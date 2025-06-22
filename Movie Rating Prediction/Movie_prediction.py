import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import re

def clean_votes(value):
    """Clean Votes column, handle commas and currency suffixes like $5.16M."""
    if pd.isnull(value):
        return np.nan
    v = str(value).replace(',', '').strip()
    if v.startswith('$'):
        v = v[1:]
        if v[-1] in ['M', 'm']:
            try:
                return float(v[:-1]) * 1_000_000
            except:
                return np.nan
        elif v[-1] in ['K', 'k']:
            try:
                return float(v[:-1]) * 1_000
            except:
                return np.nan
        else:
            try:
                return float(v)
            except:
                return np.nan
    else:
        try:
            return float(v)
        except:
            return np.nan

def convert_duration_to_mins(duration):
    """
    Converts duration to total minutes.
    Accepts strings like '2h 30m', '150 min', '150', or numeric values.
    """
    if pd.isnull(duration):
        return np.nan
    if isinstance(duration, (int, float)):
        # If already numeric, assume minutes
        return int(duration)
    duration = str(duration).lower().strip()
    # Match patterns like '2h 30m'
    pattern = re.compile(r'(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?')
    match = pattern.match(duration)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        mins = int(match.group(2)) if match.group(2) else 0
        total_mins = hours * 60 + mins
        if total_mins == 0:
            # maybe duration was just number in string form without units
            try:
                return int(float(duration))
            except:
                return np.nan
        return total_mins
    else:
        # try converting directly
        try:
            return int(float(duration))
        except:
            return np.nan

def main():
    # Load dataset
    csv_path = r"D:\\Coding\\CODSOFT\\Movie Rating Prediction\\dataset_movieprediction\\IMDb Movies India.csv"
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')

    print("Columns found:", df.columns.tolist())

    # Clean columns
    df.columns = df.columns.str.strip()

    if 'IMDB Rating' in df.columns:
        df.rename(columns={'IMDB Rating': 'Rating'}, inplace=True)

    # Drop irrelevant columns
    drop_cols = ['Name', 'Year']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Clean Votes column
    if 'Votes' in df.columns:
        df['Votes'] = df['Votes'].apply(clean_votes)
    else:
        print("Warning: 'Votes' column not found.")

    # Convert Duration to minutes integer
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(convert_duration_to_mins)
    else:
        print("Warning: 'Duration' column not found.")

    # Convert Rating to numeric
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    else:
        print("Warning: 'Rating' column not found.")

    # Fill missing categorical with 'Unknown'
    for cat_col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna('Unknown')

    # Fill missing numeric with median BEFORE split
    for num_col in ['Duration', 'Votes', 'Rating']:
        if num_col in df.columns:
            median_val = df[num_col].median()
            df[num_col] = df[num_col].fillna(median_val)

    # Visualizations

    # 1. Distribution of Ratings
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Rating'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    # 2. Distribution of Duration
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Duration'], bins=30, kde=True, color='green')
    plt.title('Distribution of Movie Duration (minutes)')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Count')
    plt.show()

    # 3. Votes distribution (log scale)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Votes'], bins=50, kde=True, color='orange')
    plt.title('Distribution of Votes')
    plt.xlabel('Votes')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.show()

    # 4. Count of movies by Genre (top 10 genres)
    plt.figure(figsize=(10, 6))
    top_genres = df['Genre'].value_counts().head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='magma')
    plt.title('Top 10 Movie Genres by Count')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.show()

    # 5. Average Rating by Director (top 10 directors by count)
    plt.figure(figsize=(10, 6))
    top_directors = df['Director'].value_counts().head(10).index
    avg_ratings = df[df['Director'].isin(top_directors)].groupby('Director')['Rating'].mean().sort_values()
    sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette='coolwarm')
    plt.title('Average Rating of Top 10 Directors')
    plt.xlabel('Average Rating')
    plt.ylabel('Director')
    plt.show()

    # 6. Correlation heatmap of numeric features
    plt.figure(figsize=(8, 6))
    corr = df[['Duration', 'Votes', 'Rating']].corr()
    sns.heatmap(corr, annot=True, cmap='Blues')
    plt.title('Correlation Heatmap (Duration, Votes, Rating)')
    plt.show()

    # Encoding categorical variables
    label_encoders = {}
    for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Prepare features and target
    features = ['Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    X = df[features]
    y = df['Rating']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15, min_samples_split=5)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features, palette='viridis', orient='h')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    # Show sample predictions
    results = X_test.copy()
    results['Actual Rating'] = y_test
    results['Predicted Rating'] = y_pred
    print("\nSample Predictions:")
    print(results.head(25))


if __name__ == "__main__":
    main()
