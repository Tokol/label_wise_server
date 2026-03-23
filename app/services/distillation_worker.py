import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from app.core.config import settings


@dataclass
class WorkerConfig:
    base_url: str
    worker_id: str
    poll_seconds: float
    once: bool = False


class DistillationWorkerClient:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.api_base = f"{config.base_url.rstrip('/')}{settings.api_prefix}"

    def _json_request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(f"{self.api_base}{path}", data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc

    def fetch_next_queued_job(self) -> dict[str, Any] | None:
        query = parse.urlencode({"status": "queued", "limit": 1, "skip": 0})
        payload = self._json_request("GET", f"/distillation-jobs?{query}")
        jobs = (payload or {}).get("jobs") or []
        return jobs[0] if jobs else None

    def claim_job(self, job_id: int) -> dict[str, Any]:
        return self._json_request("POST", f"/distillation-jobs/{job_id}/claim", {"worker_id": self.config.worker_id})

    def update_progress(
        self,
        job_id: int,
        *,
        status: str,
        progress_stage: str,
        log_message: str | None = None,
        artifact_uri: str | None = None,
        metrics_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "progress_stage": progress_stage,
        }
        if log_message:
            payload["log_message"] = log_message
        if artifact_uri:
            payload["artifact_uri"] = artifact_uri
        if metrics_json is not None:
            payload["metrics_json"] = metrics_json
        return self._json_request("PATCH", f"/distillation-jobs/{job_id}/progress", payload)

    def complete_job(
        self,
        job_id: int,
        *,
        model_name: str,
        artifact_uri: str,
        metrics_json: dict[str, Any],
        log_message: str,
    ) -> dict[str, Any]:
        return self._json_request(
            "POST",
            f"/distillation-jobs/{job_id}/complete",
            {
                "model_name": model_name,
                "artifact_uri": artifact_uri,
                "metrics_json": metrics_json,
                "log_message": log_message,
            },
        )

    def fail_job(self, job_id: int, *, error_message: str, log_message: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"error_message": error_message}
        if log_message:
            payload["log_message"] = log_message
        return self._json_request("POST", f"/distillation-jobs/{job_id}/fail", payload)

    def download_training_export(self, batch_id: str) -> list[dict[str, Any]]:
        req = request.Request(
            f"{self.api_base}/records/export-batches/{parse.quote(batch_id)}/training-export",
            headers={"Accept": "application/x-ndjson"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"GET training export failed: {exc.code} {detail}") from exc

        return [json.loads(line) for line in raw.splitlines() if line.strip()]

def _write_jsonl(path: Path, dataset: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in dataset), encoding="utf-8")


def _run_trainer(
    *,
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    base_model: str,
    task_type: str,
    train_count: int | None,
    validation_count: int | None,
) -> dict[str, Any]:
    command = [
        settings.trainer_python_bin,
        "-m",
        settings.trainer_module,
        "--input-jsonl",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--model-name",
        model_name,
        "--base-model",
        base_model,
        "--task-type",
        task_type,
        "--backend",
        os.getenv("LABEL_WISE_TRAINER_BACKEND", "artifact_only"),
    ]
    if train_count is not None:
        command.extend(["--train-count", str(train_count)])
    if validation_count is not None:
        command.extend(["--validation-count", str(validation_count)])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "trainer process failed")
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"trainer returned invalid JSON: {completed.stdout}") from exc


def _run_job(client: DistillationWorkerClient, job: dict[str, Any]) -> None:
    job_id = int(job["id"])
    batch_id = str(job["batch_id"])
    train_count = job.get("train_record_count")
    validation_count = job.get("validation_record_count")

    client.claim_job(job_id)
    print(f"[worker] claimed job #{job_id} for batch {batch_id}")

    dataset = client.download_training_export(batch_id)
    artifacts_root = Path(settings.worker_artifacts_dir)
    job_dir = artifacts_root / f"job_{job_id}"
    dataset_path = job_dir / "training_export.jsonl"
    _write_jsonl(dataset_path, dataset)

    client.update_progress(
        job_id,
        status="preparing_dataset",
        progress_stage="Downloaded JSONL export and staged trainer input",
        log_message=f"Loaded {len(dataset)} JSONL records from export batch {batch_id} into {dataset_path}.",
    )
    time.sleep(1.0)

    client.update_progress(
        job_id,
        status="training",
        progress_stage="Running trainer module for fine-tuning",
        log_message=f"Invoking {settings.trainer_module} through {settings.trainer_python_bin}.",
    )
    training_result = _run_trainer(
        dataset_path=dataset_path,
        output_dir=job_dir / "model_artifact",
        model_name=f"slm_job_{job_id}",
        base_model=str(job["base_model"]),
        task_type=str(job["task_type"]),
        train_count=train_count,
        validation_count=validation_count,
    )
    client.update_progress(
        job_id,
        status="evaluating",
        progress_stage="Trainer finished and worker is registering evaluation output",
        log_message="Trainer returned metrics and artifact bundle.",
        artifact_uri=training_result["artifact_uri"],
        metrics_json=training_result["metrics_json"],
    )
    time.sleep(1.0)

    completed = client.complete_job(
        job_id,
        model_name=training_result["model_name"],
        artifact_uri=training_result["artifact_uri"],
        metrics_json=training_result["metrics_json"],
        log_message="External worker completed trainer execution and artifact registration.",
    )
    print(f"[worker] completed job #{job_id} -> {completed.get('artifact_uri')}")


def run_worker(config: WorkerConfig) -> int:
    client = DistillationWorkerClient(config)
    print(f"[worker] polling {client.api_base} as {config.worker_id}")

    while True:
        try:
            job = client.fetch_next_queued_job()
            if job is None:
                if config.once:
                    print("[worker] no queued jobs found")
                    return 0
                time.sleep(config.poll_seconds)
                continue

            try:
                _run_job(client, job)
            except Exception as exc:
                job_id = int(job["id"])
                try:
                    client.fail_job(
                        job_id,
                        error_message=str(exc),
                        log_message=f"External worker failed while processing the job: {exc}",
                    )
                except Exception as fail_exc:
                    print(f"[worker] failed to mark job #{job_id} as failed: {fail_exc}", file=sys.stderr)
                print(f"[worker] job #{job_id} failed: {exc}", file=sys.stderr)
                if config.once:
                    return 1
        except KeyboardInterrupt:
            print("[worker] stopping")
            return 0
        except Exception as exc:
            print(f"[worker] polling error: {exc}", file=sys.stderr)
            if config.once:
                return 1
            time.sleep(config.poll_seconds)


def main() -> int:
    once = "--once" in sys.argv
    config = WorkerConfig(
        base_url=settings.worker_base_url,
        worker_id=settings.worker_id,
        poll_seconds=settings.worker_poll_seconds,
        once=once,
    )
    return run_worker(config)


if __name__ == "__main__":
    raise SystemExit(main())
