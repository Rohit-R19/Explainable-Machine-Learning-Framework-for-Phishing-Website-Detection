import pandas as pd
import os

base_path = os.path.dirname(__file__)

file_path = os.path.join(
    base_path, "..", "Data", "datafile.csv"
)

df = pd.read_csv(file_path)

print("✅ Data Loaded Successfully!\n")
print(df.head())
print("\nColumns:", df.columns)
print("\nShape:", df.shape)

target_column = "CLASS_LABEL"

print("\nTarget exists:", target_column in df.columns)

df_numeric = df.select_dtypes(include=['number'])

if target_column not in df_numeric.columns:
    df_numeric[target_column] = df[target_column]

X = df_numeric.drop(target_column, axis=1)
y = df_numeric[target_column]

if "id" in X.columns:
    X = X.drop("id", axis=1)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nTop correlations with target:\n")
print(df_numeric.corr()["CLASS_LABEL"].sort_values(ascending=False))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

models = {}

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
models["Logistic Regression"] = (lr, X_test_scaled)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
models["Random Forest"] = (rf, X_test)

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
models["XGBoost"] = (xgb, X_test)

print("\n📊 Model Comparison:\n")

accuracies = {}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name}: {acc:.4f}")

best_model_name = max(accuracies, key=accuracies.get)
best_model, best_X = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")

y_pred = best_model.predict(best_X)

print("\n🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='importance', ascending=False
).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='importance',
    y='feature',
    data=feature_importance
)

plt.title("Top 15 Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance_clean.png")
plt.close()

import shap

print("\nGenerating SHAP explanations...")

X_sample = X_test.sample(min(1000, len(X_test)))

explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("shap_summary.png")
plt.close()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

from sklearn.metrics import roc_curve, auc

if best_model_name == "Logistic Regression":
    y_probs = best_model.predict_proba(X_test_scaled)[:,1]
else:
    y_probs = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

names = [x[0] for x in sorted_models]
values = [x[1] for x in sorted_models]

plt.figure(figsize=(8,5))
sns.barplot(x=names, y=values)

plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)

plt.savefig("model_comparison.png")
plt.close()
