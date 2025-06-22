# === IRIS FLOWER CLASSIFICATION (HIGH ACCURACY VERSION) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, 
                                   cross_val_score, 
                                   StratifiedKFold,
                                   GridSearchCV)
from sklearn.preprocessing import (StandardScaler, 
                                 LabelEncoder,
                                 PowerTransformer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                            GradientBoostingClassifier,
                            StackingClassifier)
from sklearn.metrics import (classification_report, 
                           confusion_matrix, 
                           accuracy_score,
                           balanced_accuracy_score)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

# ================== SETUP & CONFIGURATION ==================
def configure_plots():
    """Configure matplotlib and seaborn settings for better visuals"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.constrained_layout.use': True,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

configure_plots()

# ================== DATA LOADING & PREPROCESSING ==================
def load_and_preprocess_data(filepath):
    """Load and preprocess the iris dataset with enhanced feature engineering"""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from: {filepath}")
        
        # Enhanced feature engineering
        df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
        df['petal_ratio'] = df['petal_length'] / df['petal_width']
        df['sepal_area'] = df['sepal_length'] * df['sepal_width']
        df['petal_area'] = df['petal_length'] * df['petal_width']
        df['size_diff'] = df['sepal_area'] - df['petal_area']
        
        # Encode target variable
        le = LabelEncoder()
        df['species_encoded'] = le.fit_transform(df['species'])
        target_names = le.classes_
        
        # Define features and target
        feature_names = ['sepal_length', 'sepal_width', 
                        'petal_length', 'petal_width',
                        'sepal_ratio', 'petal_ratio',
                        'sepal_area', 'petal_area',
                        'size_diff']
        
        return df, feature_names, target_names
    
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

# ================== VISUALIZATION FUNCTIONS ==================
def plot_advanced_distributions(df, features, target_col):
    """Create enhanced visualization of data distributions"""
    # Violin plots for better distribution understanding
    plt.figure(figsize=(14, 8))
    melted = df.melt(id_vars=target_col, value_vars=features[:4])
    sns.violinplot(data=melted, x=target_col, y='value', hue='variable',
                  split=True, inner='quartile', palette='muted')
    plt.title('Advanced Feature Distributions by Species', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Pairplot with regression lines
    g = sns.pairplot(df, hue=target_col, palette='deep', 
                    plot_kws={'alpha': 0.8, 's': 60, 'edgecolor': 'k'},
                    diag_kws={'alpha': 0.8, 'edgecolor': 'k'})
    g.map_lower(sns.regplot, scatter=False, ci=False)
    g.fig.suptitle('Feature Relationships with Regression', y=1.02)
    plt.show()

# ================== MODEL TRAINING WITH HYPERPARAMETER TUNING ==================
def get_best_model(X_train, y_train, model, param_grid):
    """Perform grid search to find optimal hyperparameters"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_enhanced_models(X_train, y_train, X_test, y_test, feature_names):
    """Train and evaluate multiple classifiers with hyperparameter tuning"""
    # Base models
    base_models = {
        'KNN': make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            KNeighborsClassifier()
        ),
        'SVM': make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            SVC(probability=True)
        ),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'KNN': {'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
               'kneighborsclassifier__weights': ['uniform', 'distance']},
        'SVM': {'svc__C': [0.1, 1, 10, 100],
               'svc__gamma': [0.001, 0.01, 0.1, 1]},
        'Decision Tree': {'max_depth': [3, 5, 7, None],
                         'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [100, 200, 300],
                         'max_depth': [5, 10, None]},
        'Gradient Boosting': {'n_estimators': [100, 150, 200],
                             'learning_rate': [0.01, 0.1, 0.2],
                             'max_depth': [3, 5, 7]}
    }
    
    # Stacking classifier
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    final_estimator = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis()
    )
    
    models = {
        **base_models,
        'Stacking': StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Hyperparameter tuning for base models
        if name in param_grids:
            model = get_best_model(X_train, y_train, model, param_grids[name])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1)
        
        # Final training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'class_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results

# ================== MAIN EXECUTION ==================
def main():
    try:
        # Load and preprocess data
        filepath = r"D:\\Coding\\CODSOFT\\Iris_flower_classification\\Dataset_Iris\\IRIS.csv"
        iris_df, feature_names, target_names = load_and_preprocess_data(filepath)
        
        # Data exploration
        print("\n=== Dataset Information ===")
        print(iris_df.info())
        print("\n=== Class Distribution ===")
        print(iris_df['species'].value_counts())
        
        # Visualize data
        plot_advanced_distributions(iris_df, feature_names, 'species')
        
        # Prepare data for modeling
        X = iris_df[feature_names].values
        y = iris_df['species_encoded'].values
        
        # Apply power transform to make data more Gaussian-like
        pt = PowerTransformer()
        X = pt.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train and evaluate models
        results = train_enhanced_models(X_train, y_train, X_test, y_test, feature_names)
        
        # Print final reports
        print("\n=== FINAL MODEL PERFORMANCE ===")
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        for name, result in results.items():
            print(f"\n🔹 {name}:")
            print(f"Test Accuracy: {result['accuracy']:.4f}")
            print(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}")
            print(f"CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            if name == best_model[0]:
                print("BEST PERFORMING MODEL")
            print("Classification Report:")
            print(classification_report(y_test, result['model'].predict(X_test), 
                                      target_names=target_names))
        
        # Feature importance analysis
        if hasattr(best_model[1]['model'], 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importances = best_model[1]['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(feature_names)), importances[indices],
                   color=sns.color_palette("viridis", len(feature_names)))
            plt.title('Feature Importances in Best Model', pad=20)
            plt.xticks(range(len(feature_names)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
    finally:
        print("\n=== Execution completed ===")

if __name__ == "__main__":
    main()