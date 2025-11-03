# 💼 Salary Prediction App

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-brightgreen?logo=streamlit)](https://predict-salary-2025.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Prototype-lightgrey)](#)

---

## 📊 Overview
This interactive **AI-powered Salary Prediction App** estimates expected salaries based on a user’s job profile — including **job title, experience, education, industry, and location**.  

It’s a complete end-to-end machine learning workflow:
- **Data cleaning & preprocessing**
- **Model training & evaluation**
- **Interactive web app (Streamlit)**
- **Deployed on Streamlit Cloud**

---

## 🚀 Live Demo
👉 [**Try the App Here**](https://predict-salary-2025.streamlit.app/)

---

## 🧠 How It Works

| Step | Description |
|------|--------------|
| **1. Data Collection** | Kaggle’s Glassdoor dataset (U.S. salary data) |
| **2. Cleaning** | Handled missing values, normalized columns, encoded categories |
| **3. Modeling** | Compared 3 regression models: Linear, Gradient Boosting, Random Forest |
| **4. Deployment** | Saved best model (`RandomForestRegressor`, R² ≈ 0.62) and built a Streamlit interface |

---

## 🧩 Features
- Input key job factors via sidebar (job title, years of experience, education, etc.)
- Predicts estimated **annual salary**
- Displays input summary and model transparency info
- Clean, responsive web design (mobile-friendly)

---

## 🧰 Tech Stack
| Category | Tools |
|-----------|-------|
| **Language** | Python |
| **Libraries** | pandas, numpy, scikit-learn, streamlit, joblib |
| **Model** | Random Forest Regressor |
| **Deployment** | Streamlit Cloud |

---

## 📂 Project Structure
SalaryPrediction/
├── app/
│ ├── app.py # Streamlit app
│ └── model.pkl # Trained ML model
├── data/
│ └── cleaned_salary_data.csv
├── notebooks/
│ ├── 01_data_cleaning_and_eda.ipynb
│ └── 02_model_training.ipynb
├── requirements.txt
└── README.md


---

## 📈 Model Performance
| Model | R² | MAE | RMSE |
|--------|----|-----|------|
| **Random Forest** | 0.62 | 16.3K | 24.8K |
| Gradient Boosting | 0.55 | 20.3K | 26.9K |
| Linear Regression | 0.21 | 28.3K | 35.8K |

---

## 🌱 Future Improvements
- Fix salary scaling (normalize to realistic USD range)
- Add "What-if" salary simulator
- Expand dataset to Asia or global markets
- Integrate skill-based prediction (e.g., “+Python” → +5% salary)

---

## 👤 Author
**Chenxuan Zhang**  
AI & Data Enthusiast | Aspiring AI Project Lead  
📍 Based in Asia | 🌐 [LinkedIn Profile](#) (add yours here)

---

## 🪄 How to Run Locally
```bash
# Clone repository
git clone https://github.com/ritazcx/predict-salary.git
cd predict-salary

# Create environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/app.py

