import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'HAM10000_features_couleurs.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Display the first few rows of the dataframe
print("Original Data Head:")
print(df.head())

# --- Data Preprocessing ---

# Encode the target variable 'dx'
le = LabelEncoder()
df['dx'] = le.fit_transform(df['dx'])

# Separate features (X) and target (y)
X = df.drop(['dx', 'image_id'], axis=1)
y = df['dx']

# Identify and encode categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection to reduce overfitting and improve generalization
print("\n--- Feature Selection ---")
selector = SelectKBest(f_classif, k=min(15, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
print(f"Selected {X_train_selected.shape[1]} best features out of {X_train_scaled.shape[1]}")

# --- Model Training with Hyperparameter Tuning ---

print("\n--- Hyperparameter Tuning with GridSearchCV ---")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize Random Forest with balanced class weights
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# Perform grid search with cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest with GridSearchCV (this may take a few minutes)...")
grid_search.fit(X_train_selected, y_train)

# Best model from grid search
best_rf_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# --- Model Evaluation ---

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test_selected)
y_pred_proba = best_rf_model.predict_proba(X_test_selected)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Test Set Performance (Tuned Random Forest) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Compare Multiple Models ---
print("\n--- Comparing Multiple Models ---")

models = {
    'Tuned Random Forest': best_rf_model,
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5),
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'SVM': SVC(random_state=42, class_weight='balanced', probability=True, kernel='rbf')
}

results = []
for name, model in models.items():
    if name != 'Tuned Random Forest':
        model.fit(X_train_selected, y_train)
    
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_sc,
        'ROC-AUC': auc
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best model based on F1 score
best_model_name = results_df.loc[results_df['F1'].idxmax(), 'Model']
print(f"\nBest performing model based on F1 score: {best_model_name}")

# --- Cross-validation for more reliable testing ---
print("\n--- Stratified K-Fold Cross-Validation (Best Model) ---")

# Prepare full dataset
X_full_scaled = scaler.fit_transform(X)
X_full_selected = selector.fit_transform(X_full_scaled, y)

# Cross-validate the best model
cv_scores = cross_val_score(best_rf_model, X_full_selected, y, cv=cv_strategy, scoring='accuracy')
cv_f1_scores = cross_val_score(best_rf_model, X_full_selected, y, cv=cv_strategy, scoring='f1')

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"Cross-validation F1 scores: {cv_f1_scores}")
print(f"Mean CV F1 score: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores):.4f})")
