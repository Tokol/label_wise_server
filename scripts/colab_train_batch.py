import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a training batch from Label Wise Server and run training locally (Colab-friendly).")
    parser.add_argument("--server-url", required=True, help="Base URL of the label_wise_server deployment, e.g. https://label-wise-server.onrender.com")
    parser.add_argument("--batch-id", required=True, help="Distillation batch id to download")
    parser.add_argument("--output-dir", required=True, help="Directory where artifacts should be written")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-name", default=None, help="Optional override for produced model version name")
    parser.add_argument("--task-type", default="overall_status_classification")
    parser.add_argument("--backend", default="artifact_only")
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
    download_batch_export(args.server_url, args.batch_id, export_path)

    result = run_training(
        input_jsonl=export_path,
        output_dir=output_dir / "model_artifact",
        model_name=args.model_name or f"colab_{args.batch_id}",
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

    print(
        json.dumps(
            {
                "batch_id": args.batch_id,
                "downloaded_export": str(export_path),
                "model_name": result.model_name,
                "artifact_uri": result.artifact_uri,
                "metrics_json": result.metrics_json,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
