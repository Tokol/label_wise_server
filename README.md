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
```

### 4. Train one exported batch directly from your server
This helper downloads the JSONL export and calls the same trainer module used by the external worker:

```bash
!python scripts/colab_train_batch.py \
  --server-url https://label-wise-server.onrender.com \
  --batch-id <your_batch_id> \
  --output-dir /content/label_wise_artifacts/<your_batch_id> \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --backend hf_peft_seqcls
```

### 5. Save artifacts
Store the produced artifact directory in Google Drive or upload it to Hugging Face Hub before the Colab session ends.

### Notes
- `artifact_only` backend keeps the flow testable without full training.
- `hf_peft_seqcls` is the real LoRA-style training path and needs the extra ML dependencies above.
- The helper script writes `downloaded_batch.jsonl` and a `model_artifact/` directory containing:
  - `metrics.json`
  - `model_info.json`
  - `training_export_snapshot.jsonl`
  - `training_config.json`
  - `artifact_manifest.txt`
