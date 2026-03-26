import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, parse, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.training_runner import run_training


def download_batch_export(server_url: str, batch_id: str, destination: Path) -> Path:
    api_url = f"{server_url.rstrip('/')}/api/records/export-batches/{parse.quote(batch_id)}/training-export"
    req = request.Request(api_url, headers={"Accept": "application/x-ndjson"}, method="GET")
    try:
        with request.urlopen(req, timeout=120) as response:
            payload = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"failed to download batch export: {exc.code} {detail}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return destination


def json_request(server_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(f"{server_url.rstrip('/')}/api{path}", data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"request failed: {method} {path}: {exc.code} {detail}") from exc

    return json.loads(raw) if raw else {}


def upload_artifact_to_hugging_face(
    *,
    artifact_dir: Path,
    repo_id: str,
    repo_subpath: str,
    token: str | None,
    commit_message: str,
) -> str:
    try:
        from huggingface_hub import upload_folder
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --hf-repo-id uploads. Install it in Colab with `pip install huggingface_hub`."
        ) from exc

    upload_folder(
        repo_id=repo_id,
        folder_path=str(artifact_dir),
        path_in_repo=repo_subpath,
        repo_type="model",
        token=token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}/tree/main/{repo_subpath}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a training batch from Label Wise Server and run training locally (Colab-friendly).")
    parser.add_argument("--server-url", required=True, help="Base URL of the label_wise_server deployment, e.g. https://label-wise-server.onrender.com")
    parser.add_argument("--job-id", type=int, default=None, help="Optional distillation job id to claim/update/complete")
    parser.add_argument("--batch-id", required=True, help="Distillation batch id to download")
    parser.add_argument("--output-dir", required=True, help="Directory where artifacts should be written")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-name", default=None, help="Optional override for produced model version name")
    parser.add_argument("--task-type", default="overall_status_classification")
    parser.add_argument("--backend", default="artifact_only")
    parser.add_argument("--worker-id", default="colab-manual", help="Worker identifier to write into job logs when job_id is provided")
    parser.add_argument("--hf-repo-id", default=None, help="Optional Hugging Face model repo id to upload the finished artifact into")
    parser.add_argument("--hf-repo-subpath", default=None, help="Optional repo subpath. Defaults to versions/<model_name>")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Falls back to HF_TOKEN env var")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    export_path = output_dir / "downloaded_batch.jsonl"
    try:
        if args.job_id is not None:
            json_request(
                args.server_url,
                "POST",
                f"/distillation-jobs/{args.job_id}/claim",
                {"worker_id": args.worker_id},
            )

        download_batch_export(args.server_url, args.batch_id, export_path)

        if args.job_id is not None:
            json_request(
                args.server_url,
                "PATCH",
                f"/distillation-jobs/{args.job_id}/progress",
                {
                    "status": "training",
                    "progress_stage": "Colab notebook is running the trainer",
                    "log_message": f"Colab downloaded batch {args.batch_id} into {export_path}.",
                },
            )

        result = run_training(
            input_jsonl=export_path,
            output_dir=output_dir / "model_artifact",
            model_name=args.model_name or (f"slm_job_{args.job_id}" if args.job_id is not None else f"colab_{args.batch_id}"),
            base_model=args.base_model,
            task_type=args.task_type,
            train_count=args.train_count,
            validation_count=args.validation_count,
            backend=args.backend,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        artifact_uri = result.artifact_uri
        if args.hf_repo_id:
            repo_subpath = args.hf_repo_subpath or f"versions/{result.model_name}"
            artifact_uri = upload_artifact_to_hugging_face(
                artifact_dir=output_dir / "model_artifact",
                repo_id=args.hf_repo_id,
                repo_subpath=repo_subpath,
                token=args.hf_token or os.getenv("HF_TOKEN"),
                commit_message=f"Upload trained artifact for {result.model_name}",
            )

        if args.job_id is not None:
            json_request(
                args.server_url,
                "PATCH",
                f"/distillation-jobs/{args.job_id}/progress",
                {
                    "status": "evaluating",
                    "progress_stage": "Colab finished training and is registering artifacts",
                    "log_message": "Trainer finished in Colab and returned metrics.",
                    "artifact_uri": artifact_uri,
                    "metrics_json": result.metrics_json,
                },
            )
            json_request(
                args.server_url,
                "POST",
                f"/distillation-jobs/{args.job_id}/complete",
                {
                    "model_name": result.model_name,
                    "artifact_uri": artifact_uri,
                    "metrics_json": result.metrics_json,
                    "log_message": "Colab reported successful training completion.",
                },
            )

        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "batch_id": args.batch_id,
                    "downloaded_export": str(export_path),
                    "model_name": result.model_name,
                    "artifact_uri": artifact_uri,
                    "metrics_json": result.metrics_json,
                }
            )
        )
        return 0
    except Exception as exc:
        if args.job_id is not None:
            try:
                json_request(
                    args.server_url,
                    "POST",
                    f"/distillation-jobs/{args.job_id}/fail",
                    {
                        "error_message": str(exc),
                        "log_message": f"Colab run failed: {exc}",
                    },
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
