# 📈 Bondora P2P Loan Regression

This project builds a regression model to analyze and predict loan-related outcomes using the **Bondora P2P loans** dataset. It applies end-to-end machine learning workflows including preprocessing, feature engineering, modeling, and evaluation.

## 📂 Dataset

* **Source**: [Kaggle - Bondora P2P Loans](https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans)
* **Description**: The dataset consists of historical loan information from the Bondora lending platform, including borrower income, employment status, loan purpose, and more.

## 🚀 Project Features

* 📌 Data Cleaning & Preprocessing
* 🔍 Feature Selection & Engineering
* 📊 Regression Modeling:

  * Linear Regression
  * Random Forest Regressor
* �� Model Evaluation Metrics:

  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
  * R² Score
* 📉 Visualizations:

  * Prediction vs. Actual plots
  * Residual distribution

## 🛠️ Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed and the following packages:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

You can install dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/imantahami/bondora-regression.git
   cd bondora-regression
   ```

2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `Untitled-1.ipynb` and run the cells in order.

## 📈 Results & Insights

* Random Forest generally outperformed Linear Regression in predictive accuracy.
* Certain features such as income, employment status, and loan purpose had significant predictive power.
* Visual analysis revealed heteroscedasticity in some residuals, prompting potential future improvements.

## 📁 Project Structure

```
bondora-regression/
│
├── data/                   # (Optional) Raw or processed data files
├── Untitled-1.ipynb        # Main notebook with full analysis
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🧠 Future Improvements

* Try advanced models like XGBoost or LightGBM
* Hyperparameter tuning with cross-validation
* Add pipeline and model export for deployment

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:

* GitHub: [@imantahami](https://github.com/imantahami)
