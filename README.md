# Global Country Info Analysis - AI/ML/DL

A comprehensive analysis of global country data using Artificial Intelligence, Machine Learning, and Deep Learning techniques.

## 📊 Dataset

The project uses the `world-data-2023.csv` dataset containing information about 196 countries with 35 features including:
- Demographics (Population, Density, Birth Rate, Life Expectancy)
- Economics (GDP, Minimum Wage, Tax Revenue)
- Geography (Land Area, Latitude, Longitude)
- Health (Infant Mortality, Maternal Mortality, Physicians per thousand)
- Education (Primary and Tertiary Enrollment)
- Environment (CO2 Emissions, Forested Area, Agricultural Land)
- And more...

## 🤖 AI/ML/DL Implementations

### Machine Learning
1. **Regression Analysis**
   - Predicts Life Expectancy using Linear Regression and Random Forest
   - Feature importance analysis to identify key factors
   - Model evaluation with MSE and R² metrics

2. **Clustering Analysis**
   - Groups countries using K-Means clustering
   - Determines optimal clusters using the Elbow method
   - PCA visualization of country clusters
   - Silhouette score for cluster quality

3. **Classification**
   - Classifies countries by development level (Low, Medium, High)
   - Uses Random Forest Classifier
   - Based on Life Expectancy and GDP indicators

### Deep Learning
- **Neural Network for GDP Prediction**
  - Multi-layer perceptron with dropout regularization
  - 128 → 64 → 32 → 16 → 1 architecture
  - Adam optimizer with MSE loss
  - Training visualization and performance metrics

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.12+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Global-Country-Info-Analysis.git
cd Global-Country-Info-Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run the Python Script
```bash
python analysis.py
```

This will execute the complete analysis pipeline:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- ML Regression Models
- ML Clustering
- ML Classification
- DL Neural Network
- Results Summary

#### Option 2: Use Jupyter Notebook (Interactive)
```bash
jupyter notebook analysis.ipynb
```

The notebook provides an interactive environment with:
- Step-by-step analysis
- Visualizations
- Detailed explanations
- Code cells you can modify and re-run

## 📈 Features

### Exploratory Data Analysis
- Dataset overview and statistics
- Missing value analysis
- Correlation heatmaps
- Distribution plots

### Machine Learning Models
- **Linear Regression**: Baseline model for life expectancy prediction
- **Random Forest Regressor**: Advanced ensemble method with feature importance
- **K-Means Clustering**: Unsupervised grouping of similar countries
- **Random Forest Classifier**: Development level categorization

### Deep Learning
- **Neural Network**: Multi-layer architecture for complex pattern recognition
- **Dropout Regularization**: Prevents overfitting
- **Training Monitoring**: Loss curves and validation metrics

## 📊 Results

The analysis provides:
- Model performance metrics (R², MSE, Accuracy, Silhouette Score)
- Feature importance rankings
- Country clusters and classifications
- Visualizations and insights

## 🛠️ Technologies Used

- **Python 3.12**: Programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning framework
- **matplotlib/seaborn**: Data visualization
- **Jupyter**: Interactive notebooks

## 📝 Project Structure

```
Global-Country-Info-Analysis/
├── world-data-2023.csv      # Dataset
├── analysis.py              # Main analysis script
├── analysis.ipynb           # Jupyter notebook
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🎯 Key Insights

The analysis reveals:
- Strong correlation between GDP and life expectancy
- Healthcare indicators (physicians, infant mortality) are crucial for longevity
- Countries naturally cluster into distinct groups based on development
- Neural networks can effectively model complex socio-economic relationships

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.