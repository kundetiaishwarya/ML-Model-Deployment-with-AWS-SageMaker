# MultiTextClassificationModelDeployment

This project demonstrates a complete pipeline for fine-tuning a **DistilBERT** model to perform **multi-class text classification** on the **UCI News Aggregator Dataset**, using **Amazon SageMaker** for training and deployment.

---

## 📂 Dataset

- **Source**: UCI News Aggregator Dataset
- **Preprocessing**: 
  - Loaded from an S3 bucket
  - Retained only `TITLE` and `CATEGORY` columns
  - Mapped categories:
    - `e` → Entertainment
    - `b` → Business
    - `t` → Science
    - `m` → Health
  - Encoded labels numerically
  - Sampled 5% of data for faster experimentation

---

## 🤖 Model

- **Model Type**: `DistilBertForSequenceClassification`
- **Library**: [Transformers](https://huggingface.co/docs/transformers/index) by HuggingFace
- **Tokenizer**: `DistilBertTokenizer`
- **Training Framework**: PyTorch
- **Trained On**: SageMaker Jupyter Notebook instance

---

## 🏗️ Training Pipeline

Implemented in `script.py`:

- Custom PyTorch `Dataset` class for text preprocessing and tokenization
- Train/Validation split using `sklearn.model_selection.train_test_split`
- Training loop with loss and accuracy metrics
- Evaluation on validation set
- Model saved to `/opt/ml/model` and uploaded to S3

---

## ☁️ Model Deployment

Deployment logic is in `MultiTextClassificationModelDeployment.ipynb`, using `sagemaker.huggingface` SDK:

```python
from sagemaker.huggingface import HuggingFaceModel

model_artifact = "s3://hugging-face-multiclass-textclassification-bucket-custombucket/output/huggingface-pytorch-training-2025-04-20-19-06-38-792/output/model.tar.gz"

huggingface_model = HuggingFaceModel(
    model_data=model_artifact,
    role=role,
    transformers_version='4.28.1',
    pytorch_version='2.0.0',
    py_version='py310',
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
