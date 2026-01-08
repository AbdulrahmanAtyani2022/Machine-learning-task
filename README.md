
## 📌 Machine Learning task

**Non-Linear Regression & Logistic Regression Analysis**

This repository contains the complete implementation and analysis for task of the *Machine Learning and Data Science
The project explores **non-linear regression techniques** and **logistic regression models**, focusing on **model complexity, overfitting, and generalization performance**.

---

## 🧠 Part 1 — Non-Linear Regression

### A) Polynomial Regression with Ridge Regularization

* Generated a synthetic dataset of 25 noisy samples:
  [
  y = \sin(5\pi x) + \epsilon,\quad \epsilon \in [-0.3, 0.3]
  ]
* Applied **degree-9 polynomial regression** with different regularization strengths:
  [
  \lambda \in {0, 0.0001, 0.001, 0.1, 0.5}
  ]

**Key Findings:**

* **λ = 0** → Severe overfitting due to excessive flexibility
* **Small λ values** → Best trade-off between bias and variance
* **Large λ values** → Over-smoothing and underfitting

✔️ *A small but non-zero λ provided the best generalization performance.*

---

### B) Radial Basis Function (RBF) Regression

Non-linear regression was implemented **without regularization**, using Gaussian RBFs with evenly spaced centers.

**Tested configurations:**

* 1 RBF
* 5 RBFs
* 10 RBFs
* 50 RBFs

**Observations:**

* **1 RBF** → Strong underfitting
* **5 RBFs** → Captures general trend but still limited
* **10 RBFs** → Best balance and smooth approximation
* **50 RBFs** → Overfitting due to excessive flexibility

✔️ *10 RBF basis functions achieved the best generalization.*

---

## 🧪 Part 2 — Logistic Regression (Customer Churn Prediction)

### Dataset & Preprocessing

* Customer churn dataset from Assignment #1
* Missing values handled using **median imputation**
* Outliers treated via **IQR-based replacement**
* Numerical features standardized using **StandardScaler**

### Data Split

* **2500** training samples
* **500** validation samples
* **500** test samples

(Stratified splitting to preserve class balance)

---

### Models Trained

* **Linear Logistic Regression**
* **Polynomial Logistic Regression** with degrees:

  * 2
  * 5
  * 9

### Evaluation Metrics

For **train / validation / test sets**:

* Accuracy
* Precision
* Recall

---

## 📊 Model Comparison & Insights

* Accuracy remained relatively stable across models, indicating good generalization.
* **Higher-degree models** showed:

  * Increased **training precision and recall**
  * Noticeable drops on validation and test sets → signs of **overfitting**
* Polynomial degree **5** achieved the best balance between complexity and performance.

---

## 📈 ROC Curve & AUC

* Best model selected based on **validation accuracy**
* ROC curve plotted on the test set
* **AUC ≈ 0.96**, indicating excellent discrimination between churn and non-churn customers

✔️ *The model maintains a high true positive rate with low false positives.*

---

## 🏆 Key Takeaways

* Regularization is essential to control overfitting in high-degree polynomial models
* RBF regression performance is highly sensitive to the number of basis functions
* Polynomial feature expansion improves classification performance up to an optimal complexity
* Validation-based model selection is critical for reliable generalization

---

## 🚀 Technologies Used

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---


Just tell me 👍
