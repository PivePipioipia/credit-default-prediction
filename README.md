# Credit Default Prediction

This project is an end-to-end machine learning application for predicting the likelihood of a customer defaulting on their credit obligations. It includes data preprocessing, model training, evaluation, and is deployed as an interactive web app using Gradio on Hugging Face Spaces.
## ✨ Project Features

- Data Preprocessing and Exploration
Cleaned and explored the dataset to understand key features, identify missing values, and visualize patterns related to credit default.

- Initial Model Building
Built and trained baseline machine learning models:

    + Logistic Regression

    + Random Forest

- Model Improvement (Feature Enhancement & Tuning)
Enhanced model performance by refining input features and experimenting with advanced algorithms:

    + XGBoost (with improved features)

    + Logistic Regression (with improved features)

    + Random Forest (with improved features)

- Model Evaluation & Summary
Assessed models using appropriate metrics and discussed key insights, strengths, and limitations of each approach.

- Public Deployment
Deployed the best-performing model as an interactive web application using Gradio on Hugging Face Spaces.

## 🚀 Demo

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/nguyen-hong-yen/credit-default-predictor)

## 🧠 Model Overview

- **Algorithm**: XGBoost
- **Target Variable**: `default` (1 = defaulted, 0 = not defaulted)
  - **Input Features** (example):
    + Credit Limit (LIMIT_BAL): 200,000
    + Gender (1 = Male, 2 = Female): 2 (Female)
    + Education Level
    + Marital Status
    + Age
    + Other relevant attributes

##  How to Run Locally

1. **Clone the repository**
git clone https://github.com/PivePipioipia/credit-default-prediction.git
cd credit-default-prediction
2. Create and activate a virtual environment (recommended)
- On Windows:

python -m venv venv
venv\Scripts\activate

- On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
3. Install the dependencies

pip install -r requirements.txt

4. Run the app locally
python app.py


## How to Use on Hugging Face Spaces
This app is publicly available on Hugging Face Spaces.
You can use it directly in your browser without installing anything.

▶️ Quick Start
Visit the app here: https://huggingface.co/spaces/nguyen-hong-yen/credit-default-predictor

Enter the required inputs in the form (e.g., Age, Limit_bal, Education Level, etc.)

Click the "Dự đoán" button

The model will return a prediction

