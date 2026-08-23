# RateMyReview

**Machine Learning Based Review Rating Prediction System**

RateMyReview analyses user reviews and predicts a corresponding **1-5 star
rating** using a hybrid machine learning architecture: a fine-tuned
**DistilBERT** transformer for high-accuracy contextual sentiment
understanding, with a lightweight **TF-IDF + Ridge Regression** fallback for
CPU-efficient, resource-constrained deployments.

## Features

- Automated sentiment-based rating prediction (1-5 stars) from raw review text.
- Dual-backend ML architecture with automatic model selection:
  - **Primary:** fine-tuned DistilBERT (HuggingFace Transformers + PyTorch).
  - **Fallback:** TF-IDF + Ridge pipeline (scikit-learn), CPU friendly.
  - **Baseline:** built-in lexicon heuristic so the API works even before any
    model is trained.
- Modular design: data processing, models, services, API and configuration
  are cleanly separated.
- RESTful API built with FastAPI (interactive docs at `/docs`).
- Batch prediction endpoint for up to 64 reviews per request.
- Training, evaluation and Docker deployment tooling included.

## Architecture

```
                    +-----------------------------------+
   review text --->|  POST /api/v1/predict             |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |  app/services/predictor.py        |
                    |  PredictorService (facade)        |
                    +-----------------+-----------------+
                                      |
              +-----------------------+------------------------+
              |                       |                        |
   +----------v----------+ +----------v-----------+ +----------v----------+
   | app/models/         | | app/models/          | | heuristic lexicon   |
   | distilbert.py       | | tfidf_ridge.py       | | (always available)  |
   | DistilBERT 5-class  | | TF-IDF + Ridge       | |                     |
   | classifier          | | regression pipeline  | |                     |
   +---------------------+ +----------------------+ +---------------------+
```

Backend selection (`RMR_MODEL_BACKEND=auto`, the default) happens at startup:

1. **distilbert** - used when torch/transformers are installed *and*
   `models/distilbert_weights/` contains a fine-tuned checkpoint.
2. **ridge** - used when `models/ridge_model.pkl` exists.
3. **heuristic** - lexicon baseline; guarantees the service degrades
   gracefully instead of failing.

The active backend is reported in every prediction response
(`model_used`), by `GET /api/v1/model/info`, and by `GET /api/v1/health`.

## Project Structure

```
RateMyReview/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Request/response models
│   ├── models/
│   │   ├── base.py            # Shared rating utilities
│   │   ├── distilbert.py      # Transformer-based model
│   │   └── tfidf_ridge.py     # Lightweight ML fallback model
│   ├── services/
│   │   ├── predictor.py       # Prediction service layer
│   │   └── preprocessing.py   # Text preprocessing utilities
│   ├── core/
│   │   ├── config.py          # Application configuration
│   │   └── model_loader.py    # Model initialization / selection
│   └── main.py                # FastAPI application entry point
├── training/
│   ├── train_distilbert.py    # Transformer model training
│   ├── train_ridge.py         # Baseline model training
│   ├── evaluation.py          # Model evaluation pipeline
│   └── data_utils.py          # Shared dataset loading
├── data/
│   ├── raw/                   # Original review datasets (sample included)
│   └── processed/             # Cleaned training data
├── models/
│   ├── distilbert_weights/    # Fine-tuned model files (after training)
│   └── ridge_model.pkl        # Trained fallback model (after training)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **Lightweight install (no transformer backend):** if you only need the API
> with the TF-IDF/Ridge backend, install just the light dependencies listed
> at the top of `requirements.txt`. The app detects missing torch/transformers
> automatically and falls back.

### 2. Train the models

The Ridge baseline trains in seconds (about 3s on the 205k-row dataset
bundled at `data/raw/dataset_sa.csv` - see the Datasets section):

```bash
python training/train_ridge.py --data data/raw/dataset_sa.csv
```

Fine-tune DistilBERT (requires torch + transformers; needs a GPU or
patience on CPU - subsample with `--max-rows` for a quick run):

```bash
python training/train_distilbert.py --data data/raw/dataset_sa.csv --max-rows 50000 --epochs 3
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Interactive documentation: <http://127.0.0.1:8000/docs>

## Datasets

### Bundled real dataset

A copy of the Flipkart product reviews dataset (`Dataset-SA`, ~205k rows with
`product_name, product_price, Rate, Review, Summary, Sentiment` columns) is
expected at `data/raw/dataset_sa.csv` (~33 MB, not tracked in git). The
loader auto-detects its `Review`/`Rate` columns case-insensitively and drops
the ~25k rows with missing review text plus corrupted rating values.

Train and evaluate on it:

```bash
python training/train_ridge.py --data data/raw/dataset_sa.csv
python training/train_distilbert.py --data data/raw/dataset_sa.csv --max-rows 50000 --epochs 2 --batch-size 16 --max-length 128
python training/evaluation.py --data data/raw/dataset_sa.csv --sample 20000
```

Measured results (20k-review random sample; the DistilBERT holdout accuracy
from its own 10k validation split during training was 95.5%):

```
backend          MAE    RMSE   exact  within1
---------------------------------------------
distilbert     0.076   0.343  93.87%   99.17%
ridge          0.079   0.349  93.62%   99.13%
heuristic      0.960   1.299  40.27%   63.80%
```

The bundled `models/distilbert_weights/` checkpoint was fine-tuned on a 50k
subsample (2 epochs, seq length 128) in about 7 minutes on an RTX 3050
laptop GPU; the Ridge model trains on the full 205k rows in about 3 seconds
on CPU.

Holdout metrics are high because the dataset's reviews are short, repetitive
phrases strongly tied to their star ratings (and duplicates overlap between
splits); expect lower accuracy on longer, free-form reviews.

### Using your own dataset

Place a CSV in `data/raw/` with a text column (`text`, `review`,
`review_text`, `content`, ...) and a rating column (`rating`, `score`,
`stars`, `label`, ...) using a 1-5 scale (0-4 labels are auto-shifted).
For example:

```csv
text,rating
"Absolutely fantastic, arrived a day early!",5
"It does the job but feels cheap.",3
"Terrible - stopped working after one use.",1
```

Then pass `--data data/raw/your_file.csv` to the training scripts.

## API Reference

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| POST   | `/api/v1/predict`     | Predict rating for one review        |
| POST   | `/api/v1/predict/batch` | Predict ratings for up to 64 reviews |
| GET    | `/api/v1/model/info`  | Active backend + backend availability |
| GET    | `/api/v1/health`      | Liveness probe incl. active backend  |
| GET    | `/health`             | Quick liveness check                 |
| GET    | `/docs`               | Interactive OpenAPI documentation    |

### Examples

```bash
# Single prediction
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely fantastic product, works flawlessly!"}'

# Response
# {
#   "rating": 5,
#   "confidence": 0.9123,
#   "model_used": "ridge",
#   "expected_rating": 4.8721,
#   "probabilities": null,
#   "processing_time_ms": 3.81
# }

# Batch prediction
curl -X POST http://127.0.0.1:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Terrible, broke after one use.", "It is okay.", "Love it!"]}'
```

## Evaluation

Compare every available backend on a labelled dataset:

```bash
python training/evaluation.py --data data/raw/sample_reviews.csv
```

Example output:

```
backend          MAE     RMSE   exact  within1
---------------------------------------------
distilbert     0.412    0.623  71.2%    94.8%
ridge          0.534    0.701  66.9%    92.5%
heuristic      0.918    1.104  44.0%    78.0%
```

## Configuration

All settings are configurable via environment variables (prefix `RMR_`) or a
`.env` file (see `.env.example`):

| Variable                     | Default                      | Description                                |
|------------------------------|------------------------------|--------------------------------------------|
| `RMR_MODEL_BACKEND`          | `auto`                       | `auto`, `distilbert`, `ridge`, `heuristic` |
| `RMR_DEVICE`                 | `auto`                       | `auto`, `cpu`, `cuda`, `mps`               |
| `RMR_MAX_SEQUENCE_LENGTH`    | `256`                        | Transformer max token length               |
| `RMR_PREDICTION_BATCH_SIZE`  | `32`                         | Transformer inference batch size           |
| `RMR_RIDGE_MODEL_PATH`       | `models/ridge_model.pkl`     | Ridge artifact location                    |
| `RMR_DISTILBERT_WEIGHTS_DIR` | `models/distilbert_weights`  | DistilBERT checkpoint directory            |
| `RMR_CORS_ORIGINS`           | `["*"]`                      | Allowed CORS origins (JSON list)           |
| `RMR_DEBUG`                  | `false`                      | Verbose logging                            |

## Docker

```bash
docker build -t ratemyreview .
docker run -p 8000:8000 ratemyreview
# then open http://127.0.0.1:8000/docs
```

The image installs CPU-only PyTorch to stay small; mount a GPU image instead
if you need CUDA inference.

## Technology Stack

- **ML:** PyTorch, HuggingFace Transformers, scikit-learn
- **NLP:** DistilBERT, TF-IDF vectorisation, text preprocessing
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Docker
