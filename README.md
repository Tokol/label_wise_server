# Label Wise Server

Python backend for Label Wise data collection, installation registration, training-payload ingestion, and future model training/export workflows.

## Planned responsibilities
- Register app installations and issue server tokens
- Accept training payloads from the mobile app
- Enforce token-based request checks
- Store raw payloads and indexed metadata
- Provide admin APIs for inspection and exports
- Support future dataset preparation and training jobs

## Suggested stack
- FastAPI
- SQLAlchemy
- PostgreSQL
- Alembic
- Pydantic

## Initial structure
- `app/api/routes`: API route modules
- `app/core`: config and security helpers
- `app/db`: DB session/base setup
- `app/models`: ORM models
- `app/schemas`: request/response schemas
- `app/services`: business logic
- `scripts`: future data/export/training utilities
- `tests`: API and service tests

## Google Colab First Run
If you want to fine-tune before setting up a dedicated GPU service, use Colab as a temporary training machine.

### 1. Open a GPU runtime
- Colab: `Runtime` -> `Change runtime type` -> `GPU`
- Prefer paid Colab if available for better GPU reliability.

### 2. Clone this repo in Colab
```bash
!git clone <your-label-wise-server-repo-url>
%cd label_wise_server
```

### 3. Install dependencies
```bash
!pip install -r requirements.txt
!pip install torch transformers peft numpy accelerate datasets sentencepiece
!pip install huggingface_hub
```

### 4. Log in to Hugging Face in Colab
If you want the finished adapter to upload directly into your Hugging Face model repo:

```bash
from huggingface_hub import notebook_login
notebook_login()
```

### 5. Train one distillation job directly from your server
This helper downloads the JSONL export, runs the same trainer module used by the external worker, and can report completion back to the job:

```bash
!python scripts/colab_train_batch.py \
  --server-url https://label-wise-server.onrender.com \
  --job-id <your_job_id> \
  --batch-id <your_batch_id> \
  --output-dir /content/label_wise_artifacts/<your_batch_id> \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --hf-repo-id IndraDThor/label-wise-qwen25-3b-lora \
  --backend hf_peft_seqcls
```

### 6. Save artifacts
If `--hf-repo-id` is provided and you logged in, the finished artifact is uploaded to Hugging Face automatically. Keep Google Drive or a zip download as backup if you want.

### Notes
- `artifact_only` backend keeps the flow testable without full training.
- `hf_peft_seqcls` is the real LoRA-style training path and needs the extra ML dependencies above.
- If `--job-id` is provided, the helper claims the job, reports progress, and completes or fails it through the server API.
- If `--hf-repo-id` is provided, the helper uploads the finished `model_artifact/` folder into that Hugging Face model repo and reports the Hugging Face URL back to the server.
- The helper script writes `downloaded_batch.jsonl` and a `model_artifact/` directory containing:
  - `metrics.json`
  - `model_info.json`
  - `training_export_snapshot.jsonl`
  - `training_config.json`
  - `artifact_manifest.txt`
