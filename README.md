# 🛡️ Fraud Detection Using Machine Learning

Machine Learning project to detect fraudulent online payment transactions using the **PaySim dataset**. Built with feature engineering, Random Forest, and Streamlit deployment.

---

## 📌 Objective

Classify transactions as:

- ✅ Genuine  
- 🚨 Fraudulent  

Focus: maximize **Recall** to reduce missed fraud cases.

---

## 📂 Dataset

**PaySim Fraud Detection Dataset**

> Dataset not included due to file size. Place inside:

```
/data/paysim.csv
```
**PaySim Fraud Detection Dataset**

Features used:

Raw:
* `step`
* `type`
* `amount`
* `oldbalanceOrg`
* `newbalanceOrig`
* `oldbalanceDest`
* `newbalanceDest`

Engineered:
* `amount_to_balance_ratio`
* `sender_drained`
* `dest_was_zero`
* `balance_error_orig`
* `balance_error_dest`
  
Target: `isFraud`

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter / Google Colab

## 🤖 Model

### Random Forest

## 📈 Evaluation Metrics

* TN = 1,270,719
* FP = 58
* FN = 4
* TP = 1,635
* Precision: 96.57%
* Recall: 99.76%
* Accuracy: 99.99%

> Recall is prioritized in fraud detection because missing fraudulent transactions can cause financial loss.

## 🖥️ App Features
* Real-time fraud prediction
* Fraud Risk Score
* Low / Medium / High alerts
* Explanation-based insights

Run locally:
```
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Author

**Ranjan Verma** | Symbiosis Institute of Technology (SIT)
