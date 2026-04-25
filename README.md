# 💳 Fraud Detection Using Machine Learning

End-to-end machine learning project to detect fraudulent online payment transactions using the **PaySim dataset**.

## 📌 Objective

Build a classification model that predicts whether a transaction is:

* ✅ Genuine
* 🚨 Fraudulent

Main focus: maximize **Recall** to reduce missed fraud cases.

## 📂 Dataset

**PaySim Fraud Detection Dataset**

Features used:

* `step`
* `type`
* `amount`
* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

Target:

* `isFraud`

## ⚙️ Workflow

```text
Data Loading
→ Data Cleaning
→ EDA
→ Encoding
→ Train-Test Split
→ Model Training
→ Evaluation
→ Final Model Selection
→ Model Serialization (.pkl)
```

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter / Google Colab

## 🤖 Models Used

* Logistic Regression
* Decision Tree
* Random Forest (Final Selected Model)
* Naive Bayes

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC

> Recall is prioritized in fraud detection because missing fraudulent transactions can cause financial loss.

## 🚀 Future Scope

* Streamlit Web App Deployment
* Hyperparameter Tuning
* Explainable AI
* Real-time Fraud Detection API
* Banking integration

## 👨‍💻 Author

**Ranjan Verma** | Symbiosis Institute of Technology (SIT)
