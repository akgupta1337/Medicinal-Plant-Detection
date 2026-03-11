# Medicinal Plant Identification

This project provides a full pipeline to train a deep learning classifier that identifies medicinal plant species from leaf images.

## Project Structure

- `dataset/` - expected dataset root (train/val/test directories)
- `model/` - saved model weights and metadata
- `train.py` - training pipeline
- `inference.py` - model loading and prediction helper
- `api.py` - FastAPI backend
- `app.py` - Streamlit frontend
- `requirements.txt` - python dependencies

## Dataset

1. Download and extract the Kaggle dataset.
   - Dataset link: https://www.kaggle.com/datasets/sharvan123/medicinal-plant/data
2. Make sure the extracted folder is moved/renamed to `dataset/raw`.
   - Example (PowerShell): `Move-Item "dataset\Medicinal plant dataset" dataset\raw`

3. After extraction, organize data into:

```
dataset/
  train/
    class1/
    class2/
    ...
  val/
    class1/
    ...
  test/
    class1/
    ...
```

> The repository includes a `prepare_dataset.py` helper to split a raw dataset into train/val/test.
>
> Run:
>
> ```bash
> python prepare_dataset.py
> ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Running the API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Running the Streamlit UI

```bash
streamlit run app.py
```

## Usage

- Upload an image via the Streamlit UI to get a prediction.
- Or send a `POST /predict` request to the FastAPI server with a file field named `file`.
