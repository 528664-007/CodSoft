# === ULTIMATE ADVERTISING SALES PREDICTOR ===
# === Step 1: Import Supercharged Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# === Step 2: Load and Turbocharge Dataset ===
print(" LOADING HYPER-DATASET...")
df = pd.read_csv("D:\\Coding\\CODSOFT\\Sales_Prediction\\Dataset_salesprediction\\advertising.csv")

#  Target Engineering - Adding polynomial interactions
df['TV_Radio'] = df['TV'] * df['Radio']
df['TV_Newspaper'] = df['TV'] * df['Newspaper']
df['Radio_Newspaper'] = df['Radio'] * df['Newspaper']
df['TV_sq'] = df['TV']**2
df['Radio_sq'] = df['Radio']**2

print("\n ENHANCED DATASET HEAD:")
print(df.head().style.background_gradient(cmap='viridis'))

print("\n DATASET STATS ON STEROIDS:")
stats = df.describe().T
stats['skew'] = df.skew()
stats['kurtosis'] = df.kurt()
print(stats.style.bar(color='#d65f5f'))

# === Step 3: Next-Level Data Visualization ===
print("\n LAUNCHING ADVANCED VISUALIZATION SUITE...")

# 3D Plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['TV'], df['Radio'], df['Sales'], c=df['Sales'], cmap='plasma', s=100)
ax.set_xlabel('TV Advertising', fontsize=12)
ax.set_ylabel('Radio Advertising', fontsize=12)
ax.set_zlabel('Sales', fontsize=12)
ax.set_title('3D ADVERTISING IMPACT MAP', fontsize=16)
plt.show()

# Advanced Pairplot with Regression
sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
plt.suptitle("ADVANCED RELATIONSHIP MATRIX", y=1.02, fontsize=16)
plt.show()

# Correlation Nuclear Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='rocket_r', vmin=-1, vmax=1, center=0, 
            annot_kws={"size": 12, "weight": "bold"}, square=True)
plt.title('MEGA CORRELATION MATRIX', pad=20, fontsize=16)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.show()

# === Step 4: Feature Engineering & Selection ===
print("\n ENGINEERING SUPER FEATURES...")
X = df.drop('Sales', axis=1)
y = df['Sales']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f" Reduced dimensions from {X_scaled.shape[1]} to {X_pca.shape[1]} using PCA")

# === Step 5: Hyper-Optimized Data Splitting ===
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.15, random_state=42)

# === Step 6: Model Thunderdome - Battle of Algorithms ===
print("\n INITIATING MODEL THUNDERDOME...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=0.1),
    'Lasso Regression': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', 
                                 solver='adam', max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae
    }

# Display results
results_df = pd.DataFrame(results).T.sort_values('R² Score', ascending=False)
print("\n MODEL PERFORMANCE LEADERBOARD:")
print(results_df.style.background_gradient(subset=['R² Score'], cmap='Greens'))

# === Step 7: Hyperparameter Tuning for Champion Model ===
print("\n DEPLOYING HYPERPARAMETER OPTIMIZATION...")

# Let's tune the top performer (XGBoost in most cases)
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), 
                         params, 
                         cv=5, 
                         scoring='r2',
                         n_jobs=-1,
                         verbose=1)

print("GRID SEARCH IN PROGRESS...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n BEST MODEL PARAMETERS: {grid_search.best_params_}")

# Final evaluation
y_pred = best_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n FINAL MODEL PERFORMANCE:")
print(f"R² Score: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")

# === Step 8: Next-Gen Visualization ===
print("\n GENERATING PREDICTION VISUALIZATIONS...")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 6))
    feat_importances = pd.Series(best_model.feature_importances_, index=[f"PC_{i}" for i in range(X_pca.shape[1])])
    feat_importances.nlargest(10).plot(kind='barh', color='darkorange')
    plt.title('TOP FEATURE IMPORTANCES', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Principal Components', fontsize=12)
    plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('ADVANCED RESIDUAL ANALYSIS', fontsize=16)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Prediction Distribution
plt.figure(figsize=(12, 6))
sns.kdeplot(y_test, label='Actual Sales', color='blue', shade=True)
sns.kdeplot(y_pred, label='Predicted Sales', color='red', shade=True)
plt.title('PREDICTION DISTRIBUTION COMPARISON', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.show()

# === Step 9: Model Deployment Ready ===
print("\n MODEL DEPLOYMENT READY!")
print(f" ACHIEVED {final_r2*100:.2f}% PREDICTION ACCURACY!")
