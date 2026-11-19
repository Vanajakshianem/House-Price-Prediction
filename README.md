<<<<<<< HEAD
# House Price Prediction Using Machine Learning

A beginner-friendly mini project using Python, Applied Mathematics, Linear Regression,
Random Forest Regression, and Streamlit.

## Project features
- Loads and cleans a house-price dataset.
- Uses area, location, bathrooms, and bedrooms as input features.
- Trains Linear Regression and Random Forest models.
- Evaluates both models using MAE, RMSE, and R².
- Saves the Random Forest pipeline and provides a Streamlit prediction interface.

## Folder structure
```text
House_Price_Prediction/
├── dataset/
│   └── Bengaluru_House_Data.csv
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements
- Python 3.9 or newer
- VS Code (recommended)

## Run in VS Code / Windows PowerShell

1. Extract the ZIP file.
2. Open the extracted `House_Price_Prediction` folder in VS Code.
3. Open Terminal -> New Terminal.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Train and evaluate the models:

   ```powershell
   python train_model.py
   ```

   This creates:
   - `house_price_model.pkl`
   - `model_results.txt`

7. Start the web app:

   ```powershell
   streamlit run app.py
   ```

8. A browser tab opens. Select a location, enter house details, and click
   **Predict House Price**.

## Dataset note
The included CSV contains **1,200 synthetic property records** generated for learning
and for verifying that the project runs end-to-end. It is not actual market data and
its predictions should not be used for real property decisions.

For a more realistic academic experiment, download the Bengaluru House Price dataset
from Kaggle:
https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data

Replace `dataset/Bengaluru_House_Data.csv` with the downloaded CSV. The training
script supports the common columns `location`, `total_sqft`, `bath`, `price`, and
either `size` (for example, "2 BHK") or `bhk`. It converts simple area ranges by
taking their midpoint and drops rows that cannot be converted.

## Applied Mathematics used
- Linear equation / regression: estimates the relationship between property features and price.
- Mean and standard deviation: summarize the dataset.
- Correlation: measures linear relationships between numerical variables.
- MAE and RMSE: quantify prediction errors.
- R²: measures the proportion of target variation explained by the model.

## Important interpretation
The dataset's `price` values are in lakhs in the commonly used Bengaluru dataset.
Model quality depends on dataset size, quality, and how representative it is.
A high R² on a small sample does not guarantee real-world accuracy.
=======
# House-Price-Prediction
House Price Prediction using Machine Learning with Python, Linear Regression and Random Forest.
>>>>>>> 8b1354717f0338450e376786d874c0107fd38e5b
