# Blue Book for Bulldozers - Kaggle Competition

## Overview

This project is based on the Kaggle competition ["Blue Book for Bulldozers"](https://www.kaggle.com/competitions/bluebook-for-bulldozers) which challenges participants to predict the auction sale price of bulldozers based on their usage, equipment type, and configuration. The goal is to create a machine learning model that can accurately estimate bulldozer values using historical auction data from 1989 through 2012.

## Dataset Description

The dataset contains historical auction sale data for bulldozers sold through auctions from 1989 to 2012. The competition provides three main files with over 400,000 records of bulldozer sales and their characteristics.

### Dataset Files:
- **Train.csv**: Historical bulldozer sales from 1989-2011 with sale prices
- **Test.csv**: Bulldozer sales from 2012 for which you need to predict prices  
- **Valid.csv**: Validation set for model evaluation

### Key Features Include:

- **Equipment Identification**: 
  - SalesID, MachineID, ModelID, datasource
  - YearMade, MfgYear (manufacturing year)
  
- **Machine Specifications**:
  - Engine horsepower, machine size, transmission type
  - Hydraulics type, blade type and width
  - Enclosure (cab type), tire size, undercarriage pad width
  
- **Auction Information**:
  - Sale date (saledate), auction location, state
  - Usage hours (UsageBand), age at sale
  
- **Equipment Configuration**:
  - Drive system, hydraulics flow, blade extension
  - Various binary indicators for equipment features

### Target Variable
- **SalePrice**: The sale price of the bulldozer at auction in USD (continuous variable)

### Dataset Size
- **Training Set**: ~400,000+ records from 1989-2011
- **Test Set**: Records from 2012 (sale prices to be predicted)
- **Features**: 50+ columns including categorical, numerical, and date features

## Data Source and Download

### Kaggle Competition Data

This project uses data from the official Kaggle competition:

1. **Visit the competition page**: [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
2. **Create a Kaggle account** if you don't have one
3. **Join the competition** by clicking "Join Competition"
4. **Download the dataset** from the Data tab:
   - `Train.csv` - Training data with SalePrice
   - `Valid.csv` - Validation data with SalePrice  
   - `Test.csv` - Test data (predict SalePrice)
   - `Machine_Appendix.csv` - Additional machine information
   - `MedianHousePricing.csv` - Economic data by zip code

### Using Kaggle API (Alternative)

```bash
# Install Kaggle API
pip install kaggle

# Download competition data (requires API key setup)
kaggle competitions download -c bluebook-for-bulldozers

# Extract files
unzip bluebook-for-bulldozers.zip -d data/
```

## Project Structure

```
blue-book-bulldozers/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── bulldozer_analysis.ipynb  # Main Jupyter notebook
├── data/                  # Data directory
│   ├── Train.csv          # Training data (1989-2011)
│   ├── Valid.csv          # Validation data
│   ├── Test.csv           # Test data (2012)
│   ├── Machine_Appendix.csv  # Additional machine specs
│   └── MedianHousePricing.csv # Economic indicators
├── models/                # Saved model files
├── results/               # Prediction outputs
├── submissions/           # Kaggle submission files
└── utils/                 # Helper functions
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
# Clone the repository
git clone <https://github.com/danialasim/AI-ML-DS-Bootcamp>
cd blue-book-bulldozers

# Create virtual environment (recommended)
python -m venv bulldozer_env
source bulldozer_env/bin/activate  # On Windows: bulldozer_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Required Python Packages

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
plotly>=5.0.0
xgboost>=1.6.0
lightgbm>=3.3.0
```

## Data Preprocessing Challenges

This dataset presents several common real-world challenges:

- **Missing Values**: Many features have missing data that needs handling
- **Categorical Variables**: String categories requiring encoding
- **Date Features**: Timestamps that need feature engineering
- **Skewed Distributions**: Sale prices often follow log-normal distributions
- **High Cardinality**: Some categorical features have many unique values
- **Mixed Data Types**: Combination of numerical, categorical, and temporal data

## Modeling Approach

The project explores various machine learning algorithms:

1. **Baseline Models**: Linear Regression, Ridge, Lasso
2. **Tree-Based Models**: Random Forest, XGBoost, LightGBM
3. **Ensemble Methods**: Voting, Stacking
4. **Feature Engineering**: Date decomposition, categorical encoding, scaling

## Evaluation Metrics

- **Primary Metric**: Root Mean Squared Log Error (RMSLE) - Official Kaggle metric
- **Secondary Metrics**: RMSE, MAE, R²
- **Cross-Validation**: Time-based splits (important due to temporal nature)
- **Leaderboard**: Public/Private split based on different time periods

The competition uses RMSLE because:
- It penalizes underestimation more than overestimation
- It's scale-invariant (works well with varying price ranges)
- It handles the log-normal distribution of auction prices

## Usage

1. **Data Exploration**: Start with the exploratory data analysis section
2. **Preprocessing**: Run data cleaning and feature engineering steps
3. **Model Training**: Train various models and compare performance
4. **Prediction**: Generate predictions for test set
5. **Submission**: Format results for competition submission

```python
# Example usage
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load data
train_df = pd.read_csv('data/Train.csv', parse_dates=['saledate'])
valid_df = pd.read_csv('data/Valid.csv', parse_dates=['saledate'])
test_df = pd.read_csv('data/Test.csv', parse_dates=['saledate'])

# RMSLE evaluation function
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

# Run the notebook for complete pipeline
# jupyter notebook bulldozer_analysis.ipynb
```

## Key Insights and Features

- **Temporal Patterns**: Equipment values depreciate over time
- **Seasonal Effects**: Auction timing affects prices
- **Brand Premium**: Certain manufacturers command higher prices
- **Usage Impact**: Machine hours significantly affect valuation
- **Economic Factors**: Market conditions influence auction results

## Competition Strategy

1. **Temporal Analysis**: Understand price trends from 1989-2012
2. **Feature Engineering**: Extract meaningful features from dates and categories
3. **Missing Data Strategy**: Handle missing values systematically (70%+ missing in some columns)
4. **Time-Based Validation**: Use 2011 data to validate models predicting 2012 prices
5. **Ensemble Methods**: Combine multiple models for robust predictions
6. **Economic Context**: Incorporate economic indicators and seasonal patterns

## Submission Format

Create a CSV file with SalesID and predicted SalePrice:
```csv
SalesID,SalePrice
1139246,66000.0
1139248,57000.0
1139249,85000.0
```

## Results and Performance

- **Best Model**: [Update with your best performing model]
- **Cross-Validation Score**: [Update with CV performance]
- **Test Set Performance**: [Update with final results]
- **Feature Importance**: [Highlight most predictive features]

## Contributing

Feel free to contribute by:
- Improving feature engineering techniques
- Testing new algorithms
- Enhancing data visualization
- Optimizing model performance
- Adding documentation

## License

This project is for educational and competition purposes. Please respect the original data source terms of use.

## Contact

[Your Name] - [Your Email]
Project Link: [Your GitHub Repository URL]

---

**Note**: This is a machine learning competition project focused on regression analysis using scikit-learn and related libraries. The goal is to accurately predict bulldozer auction prices based on equipment characteristics and market conditions.