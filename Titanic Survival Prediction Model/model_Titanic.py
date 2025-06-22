import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === Step 1: Load Dataset ===
csv_path = "D:\\Coding\\CODSOFT\\Titanic Survival Prediction Model\\Dataset_Titanic\\Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

# === Step 2: Drop Unnecessary Columns ===
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# === Step 3: Handle Missing Data ===
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# === Step 4: Encode Categorical Columns ===
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])        # male=1, female=0
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])  # C=0, Q=1, S=2

# === Step 5: Split Features and Target ===
X = df.drop("Survived", axis=1)
y = df["Survived"]

# === Step 6: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Train the Model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Step 8: Make Predictions ===
y_pred = model.predict(X_test)

# === Step 9: Evaluate Model ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# === Step 10: Feature Importance Visualization ===
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.show()

# === Step 11: Survival Summary Counts ===
survived_count = y_test.sum()
not_survived_count = len(y_test) - survived_count

print(f"\nTotal Test Passengers: {len(y_test)}")
print(f"[✔] Survived: {survived_count}")
print(f"[✘] Did Not Survive: {not_survived_count}")

# === Step 12: Bar Chart of Actual Survival ===
plt.figure(figsize=(6, 4))
y_test.value_counts().plot(kind='bar', color=['salmon', 'skyblue'])
plt.title("Actual Survival Counts")
plt.xticks(ticks=[0, 1], labels=["Did Not Survive", "Survived"], rotation=0)
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# === Step 13: Display Individual Predictions (first 10) ===
results = X_test.copy()
results["Actual Survived"] = y_test.values
results["Predicted Survived"] = y_pred
print("\nFirst 50 Prediction Results:\n")
print(results.head(50))
