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

## Student Inference API
`label_wise_lite` should call the server with structured JSON instead of sending a free-form prompt and extracting braces from model text.

### Why this contract
- The current student training path is `hf_peft_seqcls`, which is an overall-status classifier.
- The server should own the response schema so Flutter does not depend on generated JSON text.
- The future inference runtime will load:
  - base model `Qwen/Qwen2.5-3B-Instruct`
  - the active LoRA adapter artifact from Hugging Face

### 1. Get active model metadata
Use this to verify which model version is currently selected for inference.

```bash
curl https://label-wise-server.onrender.com/api/student-inference/active-model
```

Example response:
```json
{
  "model_version_id": 3,
  "model_name": "slm_job_6",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "artifact_uri": "https://huggingface.co/IndraDThor/label-wise-qwen25-3b-lora/tree/main/versions/slm_job_6",
  "metrics_json": {
    "execution_mode": "hf_peft_seqcls",
    "dataset_summary": {
      "record_count": 3,
      "train": 2,
      "validation": 1
    },
    "evaluation": {
      "status_accuracy": 1.0,
      "macro_f1": 0.2,
      "eval_loss": 0.5352
    }
  }
}
```

### 2. Predict with the active student model
`label_wise_lite` should send a structured product payload.

```bash
curl -X POST https://label-wise-server.onrender.com/api/student-inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "product_name": "Instant noodles",
      "brand": "Example",
      "category": "snacks",
      "ingredients": ["wheat flour", "palm oil", "gelatin"]
    },
    "preferences": {
      "religion": {"halal": true},
      "ethical": {},
      "medical": {},
      "lifestyle": {}
    }
  }'
```

Example response:
```json
{
  "model_version_id": 3,
  "model_name": "slm_job_6",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "artifact_uri": "https://huggingface.co/IndraDThor/label-wise-qwen25-3b-lora/tree/main/versions/slm_job_6",
  "overall_status": "unsafe",
  "decision_line": "Potential conflict detected.",
  "confidence": 0.88
}
```

### Current state vs future runtime
- Right now `/api/student-inference/predict` is still simulated server logic.
- The public contract is already the right shape for `label_wise_lite`.
- The future real runtime should keep the same request and response schema while replacing the simulated logic with:
  - active model lookup
  - base Qwen load
  - active LoRA adapter load
  - model inference
