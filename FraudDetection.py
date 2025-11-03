# For data handling
import pandas as pd

# For data scaling and splitting
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# For handling the imbalanced data
from imblearn.over_sampling import SMOTE

# The machine learning model
from sklearn.ensemble import RandomForestClassifier

# For evaluating the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# Get a quick summary of the data (column types, non-null counts)
print("\n--- Data Info ---")
df.info()

# --- The MOST IMPORTANT Step: Check for Imbalance ---
# We need to see how many 'Fraud' (1) vs 'Not Fraud' (0) transactions we have.

print("\n--- Class Distribution ---")
class_counts = df['Class'].value_counts()
print(class_counts)

# Print as a percentage
print(f"\nPercentage of 'Not Fraud' (0): { (class_counts[0] / len(df)) * 100 :.2f}%")
print(f"Percentage of 'Fraud' (1):     { (class_counts[1] / len(df)) * 100 :.3f}%")

# Create a StandardScaler object
scaler = StandardScaler()

# 'Fit' the scaler to the Amount and Time columns and 'transform' them
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original Time and Amount columns
df_processed = df.drop(['Time', 'Amount'], axis=1)

print("--- Processed Data (First 5 Rows) ---")
print(df_processed.head())

# Define features (X) and target (y)
X = df_processed.drop('Class', axis=1)
y = df_processed['Class']

# Split the data
# We use stratify=y to make sure our train and test sets have the
# same percentage of fraud cases as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train (features): {X_train.shape}")
print(f"Shape of X_test (features):  {X_test.shape}")
print(f"Shape of y_train (target): {y_train.shape}")
print(f"Shape of y_test (target):  {y_test.shape}")

print("--- Before SMOTE ---")
print(f"Class 1 (Fraud) count in training data: {sum(y_train == 1)}")
print(f"Class 0 (Not Fraud) count in training data: {sum(y_train == 0)}")

# Initialize SMOTE
smote = SMOTE(random_state=42)

# 'fit_resample' creates the new, balanced dataset
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- After SMOTE ---")
print(f"Resampled Class 1 (Fraud) count: {sum(y_train_resampled == 1)}")
print(f"Resampled Class 0 (Not Fraud) count: {sum(y_train_resampled == 0)}")

# 1. Initialize the Random Forest Classifier
# n_estimators=100 means it will build 100 "decision trees"
# random_state=42 ensures you get the same results every time you run it
# n_jobs=-1 uses all your computer's processors to speed up training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Starting model training... (This may take a minute or two)")

# 2. Train the model using our balanced data
rf_model.fit(X_train_resampled, y_train_resampled)

print("Model training finished!")

# Use the trained model to make predictions on the test set
y_pred = rf_model.predict(X_test)

print("Predictions made on the test set.")

# --- The Classification Report (The most important output!) ---
# This shows Precision, Recall, and F1-score for both classes
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- The Confusion Matrix ---
# This shows us the raw numbers
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Let's visualize the confusion matrix to make it easier to read
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not Fraud (0)', 'Predicted Fraud (1)'],
            yticklabels=['Actual Not Fraud (0)', 'Actual Fraud (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Get the model's predicted probability of being 'fraud'
# This gives us a 'Risk Score' from 0.0 to 1.0
y_proba_fraud = rf_model.predict_proba(X_test)[:, 1]

# Create a new DataFrame with all our test results
results_df = X_test.copy()
results_df['Actual_Class'] = y_test
results_df['Predicted_Class'] = y_pred
results_df['Fraud_Probability'] = y_proba_fraud

# Let's add an easy-to-read 'Risk Level'
def assign_risk(prob):
    if prob > 0.8:
        return "High Risk"
    elif prob > 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"

results_df['Risk_Level'] = results_df['Fraud_Probability'].apply(assign_risk)

# Add back the original (unscaled) Amount column for better insights
# We need to find the original 'Amount' from 'df' that matches the test set
results_df = results_df.join(df['Amount']) 

print("--- Final Results DataFrame (First 5 Rows) ---")
print(results_df.head())

# --- Export to CSV ---
results_df.to_csv('fraud_detection_power_bi_results.csv', index=False)

print("\nSuccessfully exported results to 'fraud_detection_power_bi_results.csv'")
print("You are now ready to move to Power BI!")