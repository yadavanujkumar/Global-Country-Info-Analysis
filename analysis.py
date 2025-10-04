"""
Global Country Info Analysis - AI/ML/DL Implementation
This script performs comprehensive analysis using:
- Exploratory Data Analysis (EDA)
- Machine Learning (Regression, Classification, Clustering)
- Deep Learning (Neural Networks)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, classification_report, silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# For Deep Learning
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    DL_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not available. Deep Learning models will be skipped.")
    DL_AVAILABLE = False


class GlobalCountryAnalysis:
    """Comprehensive AI/ML/DL Analysis for Global Country Data"""
    
    def __init__(self, csv_file='world-data-2023.csv'):
        """Initialize the analysis with dataset"""
        self.df = pd.read_csv(csv_file)
        self.processed_df = None
        self.scaler = StandardScaler()
        print(f"Dataset loaded: {self.df.shape[0]} countries, {self.df.shape[1]} features")
        
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
        # Missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        }).sort_values('Missing', ascending=False)
        print(missing_df[missing_df['Missing'] > 0])
        
        # Basic statistics for numeric columns
        print("\nBasic Statistics:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the data for ML/DL"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Create a copy
        df = self.df.copy()
        
        # Convert percentage strings to floats
        percentage_cols = [col for col in df.columns if '%' in str(col)]
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean currency columns (remove $ and commas)
        currency_cols = ['Gasoline Price', 'GDP', 'Minimum wage']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean numeric columns with commas
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Select only numeric columns for ML
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.processed_df = df[numeric_cols].copy()
        
        # Fill missing values with median
        self.processed_df = self.processed_df.fillna(self.processed_df.median())
        
        print(f"Processed data shape: {self.processed_df.shape}")
        print(f"Numeric features: {len(numeric_cols)}")
        
        return self.processed_df
    
    def ml_regression_life_expectancy(self):
        """ML Task 1: Predict Life Expectancy using Multiple Regression"""
        print("\n" + "="*60)
        print("MACHINE LEARNING - REGRESSION (Life Expectancy Prediction)")
        print("="*60)
        
        if 'Life expectancy' not in self.processed_df.columns:
            print("Life expectancy column not found")
            return
        
        # Prepare features and target
        target = 'Life expectancy'
        df = self.processed_df.dropna(subset=[target])
        
        # Select relevant features
        features = [col for col in df.columns if col != target]
        X = df[features]
        y = df[target]
        
        # Remove columns with all NaN
        X = X.dropna(axis=1, how='all')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Linear Regression
        print("\n1. Linear Regression:")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        print(f"   MSE: {lr_mse:.2f}")
        print(f"   R² Score: {lr_r2:.4f}")
        
        # Random Forest Regression
        print("\n2. Random Forest Regression:")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"   MSE: {rf_mse:.2f}")
        print(f"   R² Score: {rf_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        print("\n3. Top 10 Important Features:")
        print(feature_importance)
        
        return {
            'lr_r2': lr_r2,
            'rf_r2': rf_r2,
            'feature_importance': feature_importance
        }
    
    def ml_clustering_countries(self):
        """ML Task 2: Cluster Countries using K-Means"""
        print("\n" + "="*60)
        print("MACHINE LEARNING - CLUSTERING (Country Grouping)")
        print("="*60)
        
        # Select features for clustering
        df = self.processed_df.copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        # Determine optimal number of clusters using elbow method
        print("\nFinding optimal number of clusters...")
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform K-Means with 4 clusters
        optimal_k = 4
        print(f"\nPerforming K-Means clustering with {optimal_k} clusters...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, clusters)
        print(f"Silhouette Score: {silhouette:.4f}")
        
        # Cluster distribution
        print("\nCluster Distribution:")
        unique, counts = np.unique(clusters, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} countries")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        print(f"\nPCA Explained Variance: {sum(pca.explained_variance_ratio_):.2%}")
        
        return {
            'clusters': clusters,
            'silhouette': silhouette,
            'pca_components': X_pca,
            'inertias': inertias
        }
    
    def ml_classification_development(self):
        """ML Task 3: Classify Countries by Development Level"""
        print("\n" + "="*60)
        print("MACHINE LEARNING - CLASSIFICATION (Development Level)")
        print("="*60)
        
        if 'Life expectancy' not in self.processed_df.columns or 'GDP' not in self.processed_df.columns:
            print("Required columns not found")
            return
        
        # Create development categories based on Life Expectancy and GDP
        df = self.processed_df.copy()
        df = df.dropna(subset=['Life expectancy', 'GDP'])
        
        # Create target variable (Development Level)
        # High: Life expectancy > 75 and GDP > median
        # Medium: Between
        # Low: Life expectancy < 65 or GDP < 25th percentile
        life_high = df['Life expectancy'].quantile(0.75)
        life_low = df['Life expectancy'].quantile(0.25)
        gdp_high = df['GDP'].quantile(0.75)
        gdp_low = df['GDP'].quantile(0.25)
        
        def classify_development(row):
            if row['Life expectancy'] >= life_high and row['GDP'] >= gdp_high:
                return 2  # High Development
            elif row['Life expectancy'] >= life_low and row['GDP'] >= gdp_low:
                return 1  # Medium Development
            else:
                return 0  # Low Development
        
        df['Development_Level'] = df.apply(classify_development, axis=1)
        
        # Prepare features and target
        target = 'Development_Level'
        features = [col for col in df.columns if col != target]
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest Classification
        print("\nRandom Forest Classifier:")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train_scaled, y_train)
        rf_pred = rf_clf.predict(X_test_scaled)
        
        print("\nClassification Report:")
        print(classification_report(y_test, rf_pred, target_names=['Low', 'Medium', 'High']))
        
        print("\nDevelopment Level Distribution:")
        print(f"  Low: {sum(y == 0)} countries")
        print(f"  Medium: {sum(y == 1)} countries")
        print(f"  High: {sum(y == 2)} countries")
        
        return {
            'accuracy': rf_clf.score(X_test_scaled, y_test),
            'model': rf_clf
        }
    
    def dl_neural_network_gdp(self):
        """Deep Learning: Neural Network for GDP Prediction"""
        print("\n" + "="*60)
        print("DEEP LEARNING - NEURAL NETWORK (GDP Prediction)")
        print("="*60)
        
        if not DL_AVAILABLE:
            print("TensorFlow/Keras not available. Skipping Deep Learning.")
            return
        
        if 'GDP' not in self.processed_df.columns:
            print("GDP column not found")
            return
        
        # Prepare features and target
        target = 'GDP'
        df = self.processed_df.dropna(subset=[target])
        
        features = [col for col in df.columns if col != target]
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features and target
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # Build Neural Network
        print("\nBuilding Neural Network...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("\nTraining Neural Network...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        print("\nEvaluating Neural Network...")
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.2e}")
        print(f"R² Score: {r2:.4f}")
        
        print(f"\nFinal Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        
        return {
            'model': model,
            'history': history,
            'r2_score': r2,
            'mse': mse
        }
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        sns.set_style('whitegrid')
        
        print("\nVisualizations would be generated here.")
        print("Note: In a full implementation, this would create:")
        print("  - Correlation heatmaps")
        print("  - Distribution plots")
        print("  - Cluster visualizations")
        print("  - Feature importance plots")
        print("  - Neural network training curves")
        print("  - Geographic maps")
    
    def run_complete_analysis(self):
        """Run all analysis steps"""
        print("\n" + "="*80)
        print("GLOBAL COUNTRY INFO ANALYSIS - AI/ML/DL")
        print("="*80)
        
        # Step 1: EDA
        self.exploratory_data_analysis()
        
        # Step 2: Preprocessing
        self.preprocess_data()
        
        # Step 3: ML - Regression
        regression_results = self.ml_regression_life_expectancy()
        
        # Step 4: ML - Clustering
        clustering_results = self.ml_clustering_countries()
        
        # Step 5: ML - Classification
        classification_results = self.ml_classification_development()
        
        # Step 6: DL - Neural Network
        dl_results = self.dl_neural_network_gdp()
        
        # Step 7: Visualizations
        self.generate_visualizations()
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nSummary:")
        if regression_results:
            print(f"  Regression R² Score: {regression_results['rf_r2']:.4f}")
        if clustering_results:
            print(f"  Clustering Silhouette Score: {clustering_results['silhouette']:.4f}")
        if classification_results:
            print(f"  Classification Accuracy: {classification_results['accuracy']:.4f}")
        if dl_results:
            print(f"  Deep Learning R² Score: {dl_results['r2_score']:.4f}")


if __name__ == "__main__":
    # Run the complete analysis
    analysis = GlobalCountryAnalysis('world-data-2023.csv')
    analysis.run_complete_analysis()
