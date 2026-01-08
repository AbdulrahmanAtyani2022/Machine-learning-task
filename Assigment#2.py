import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)



# ============================================================
# PART A — Polynomial + Ridge Regression
# ============================================================

# Generate dataset S
seed = np.random.randint(0, 100000)
np.random.seed(seed)
x = np.random.uniform(0, 1, 25)
y = np.sin(5 * np.pi * x) + np.random.uniform(-0.3, 0.3, 25)
print("Using seed:", seed)

# Sort for clean plotting
idx = np.argsort(x)
x = x[idx]
y = y[idx]

# Polynomial degree
degree = 9

# Polynomial feature matrix for training
X = np.array([x**j for j in range(degree + 1)]).T

# Smooth grid for plotting
x_plot = np.linspace(0, 1, 300)
X_plot = np.array([x_plot**j for j in range(degree + 1)]).T

# Lambdas
lambdas = [0, 0.0001, 0.001, 0.1, 0.5]

plt.figure(figsize=(8,5))
plt.scatter(x, y, label='Training Data', color='black')

for lam in lambdas:
    model = Ridge(alpha=lam, fit_intercept=False)
    model.fit(X, y)
    y_pred_plot = model.predict(X_plot)
    plt.plot(x_plot, y_pred_plot, label=f'λ = {lam}')

plt.title("Polynomial Regression with Ridge Regularization")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# PART B — RBF REGRESSION (using SAME dataset S)
# ============================================================

# RBF function
def rbf(x, center, lam):
    return np.exp(- ((x - center)**2) / lam)

# Build Φ matrix
def build_phi(x, centers, lam):
    Phi = np.zeros((len(x), len(centers)))
    for j, c in enumerate(centers):
        Phi[:, j] = rbf(x, c, lam)
    return Phi

# Solve weights: w = (ΦᵀΦ)^(-1) Φᵀ y
def solve_weights(Phi, y):
    return np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y

# Predict output
def predict(x_grid, centers, lam, w):
    return build_phi(x_grid, centers, lam) @ w

# Try 1, 5, 10, 50 RBF basis functions
basis_counts = [1, 5, 10, 50]

for M in basis_counts:
    centers = np.linspace(0, 1, M)
    lam = 1 / M   # Reasonable λ that scales with the number of basis functions

    # Build Phi and solve
    Phi = build_phi(x, centers, lam)
    w = solve_weights(Phi, y)

    # Predict on smooth grid
    y_pred = predict(x_plot, centers, lam, w)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Training Data', color='blue')
    plt.plot(x_plot, np.sin(5*np.pi*x_plot), 'k--', label='True Function: sin(5πx)')
    plt.plot(x_plot, y_pred, label=f'RBF Regression ({M} centers)')

    plt.title(f"RBF Regression Using {M} Basis Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()





# ============================================================
# PART 2 — Logestic REGRESSION
# ============================================================

warnings.filterwarnings('ignore')
# Load the dataset
df = pd.read_csv('customer_data.csv')## dataset from pervious assigment#1


# Data preprocessing for modeling with outlier handling
def preprocess_data(df):
    df_clean = df.copy()

    # Handle missing values first
    numeric_cols = ['Age', 'Income', 'Tenure', 'SupportCalls']
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Handle outliers using IQR method
    print("\n=== OUTLIER HANDLING ===")
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR

        outliers_count = ((df_clean[col] < lower_whisker) | (df_clean[col] > upper_whisker)).sum()

        if outliers_count > 0:
            median_val = df_clean[col].median()
            df_clean[col] = np.where(
                (df_clean[col] < lower_whisker) | (df_clean[col] > upper_whisker),
                median_val,
                df_clean[col]
            )
            print(f"{col}: Replaced {outliers_count} outliers with median {median_val:.2f}")

    return df_clean
# Feature scaling function
def scale_features(df, features_to_scale):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

    print("\n=== FEATURE SCALING ===")
    for feature in features_to_scale:
        print(f"{feature}: mean={df_scaled[feature].mean():.6f}, std={df_scaled[feature].std():.6f}")

    return df_scaled, scaler
# Preprocess the data with outlier handling
df_clean = preprocess_data(df)

# Scale features for machine learning
features_to_scale = ['Age', 'Income', 'Tenure', 'SupportCalls']
df_scaled, scaler = scale_features(df_clean, features_to_scale)



# 1) SPLIT: 2500 train, 500 val, 500 test


X = df_scaled[['Age', 'Gender', 'Income', 'Tenure', 'ProductType', 'SupportCalls']]
y = df_scaled['ChurnStatus']


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=2500, random_state=42, shuffle=True, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=500, random_state=42, shuffle=True, stratify=y_temp
)

print("Training size:   ", len(X_train))
print("Validation size: ", len(X_val))
print("Test size:       ", len(X_test))



# 2)evaluate model on train/val/test

def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, poly=None):

    # Transform features if polynomial model
    if poly is not None:
        X_train_ = poly.fit_transform(X_train)
        X_val_   = poly.transform(X_val)
        X_test_  = poly.transform(X_test)
    else:
        X_train_, X_val_, X_test_ = X_train, X_val, X_test

    # Train model
    model.fit(X_train_, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_)
    y_val_pred   = model.predict(X_val_)
    y_test_pred  = model.predict(X_test_)

    # Metrics
    results = {
        "name": name,
        "model": model,
        "poly": poly,

        "train_accuracy":  accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall":    recall_score(y_train, y_train_pred),

        "val_accuracy":    accuracy_score(y_val, y_val_pred),
        "val_precision":   precision_score(y_val, y_val_pred),
        "val_recall":      recall_score(y_val, y_val_pred),

        "test_accuracy":   accuracy_score(y_test, y_test_pred),
        "test_precision":  precision_score(y_test, y_test_pred),
        "test_recall":     recall_score(y_test, y_test_pred),
    }

    # Print nicely
    print(f"\n===== {name} =====")
    print("TRAIN: "
          f"Acc={results['train_accuracy']:.4f}, "
          f"Prec={results['train_precision']:.4f}, "
          f"Rec={results['train_recall']:.4f}")
    print("VAL  : "
          f"Acc={results['val_accuracy']:.4f}, "
          f"Prec={results['val_precision']:.4f}, "
          f"Rec={results['val_recall']:.4f}")
    print("TEST : "
          f"Acc={results['test_accuracy']:.4f}, "
          f"Prec={results['test_precision']:.4f}, "
          f"Rec={results['test_recall']:.4f}")


    return results



# 3) TRAIN ALL MODELS (linear + polynomial 2,5,9)

all_results = []

# 3.1 Linear Logistic Regression
linear_model = LogisticRegression(max_iter=1000)
res_linear = evaluate_model(
    "Linear Logistic Regression",
    linear_model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    poly=None
)
all_results.append(res_linear)

# 3.2 Polynomial Logistic Regression (degrees 2, 5, 9)
for degree in [2, 5, 9]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_model = LogisticRegression(max_iter=1000)

    res_poly = evaluate_model(
        f"Polynomial Logistic Regression (degree={degree})",
        poly_model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        poly=poly
    )
    all_results.append(res_poly)



# 4) CHOOSE BEST MODEL BASED ON VALIDATION PERFORMANCE

best_model_info = max(all_results, key=lambda r: r["val_accuracy"])

print("\n\n=== BEST MODEL (by validation accuracy) ===")
print("Name:", best_model_info["name"])
print(f"Validation Accuracy: {best_model_info['val_accuracy']:.4f}")

best_model = best_model_info["model"]
best_poly  = best_model_info["poly"]


if best_poly is not None:
    X_test_best = best_poly.fit_transform(X_train)  # fit on train
    X_test_best = best_poly.transform(X_test)
else:
    X_test_best = X_test

# 5) ROC CURVE + AUC FOR BEST MODEL

y_test_proba = best_model.predict_proba(X_test_best)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
auc_score = auc(fpr, tpr)

# Print AUC clearly
print("========================================")
print("          ROC / AUC RESULTS")
print(f"Best Model: {best_model_info['name']}")
print(f"AUC Score: {auc_score:.4f}")


# Save ROC figure
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_info['name']}")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()
# 6) PLOT METRICS (Accuracy, Precision, Recall) FOR TRAIN / VAL / TEST


models = [r["name"] for r in all_results]
x = np.arange(len(models))  # positions for models
width = 0.25  # bar width

metrics = ["accuracy", "precision", "recall"]

for metric in metrics:
    train_vals = [r[f"train_{metric}"] for r in all_results]
    val_vals   = [r[f"val_{metric}"]   for r in all_results]
    test_vals  = [r[f"test_{metric}"]  for r in all_results]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, train_vals, width, label="Train")
    plt.bar(x,         val_vals,   width, label="Validation")
    plt.bar(x + width, test_vals,  width, label="Test")

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} by Dataset Split for Each Model")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
