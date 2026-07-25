# Brain Tumor MRI Classifier

MobileNetV2 transfer-learning model for **4-class MRI classification**:
`glioma` · `meningioma` · `pituitary` · `notumor`

**Test accuracy: ~90.8%**

## Quick start (web app)

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** — upload an MRI image to get tumor type + confidence.

## Project layout

```
├── app.py              # Flask localhost UI
├── src/                # train / evaluate / inference
├── models/             # trained .keras weights
├── templates/ + static/
├── notebooks/
└── requirements.txt
```

## Dataset (local)

Put Kaggle [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) as:

```
brain 1/train/<class>/
brain 1/test/<class>/
```

(`brain 1/` is gitignored — too large for GitHub.)

## Train / evaluate / predict (CLI)

```bash
python src/main.py info
python src/main.py train
python src/main.py evaluate
python src/main.py predict path/to/mri.jpg
```
