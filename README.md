Retail Sales Analysis & Sales Prediction

📌 Project Overview

This project performs an end-to-end Retail Sales Analysis using Python. It covers data loading, data quality checks, exploratory analysis, visualization, customer and product analysis, correlation analysis, and machine learning-based sales amount prediction.

The notebook analyzes retail transaction data and generates visualizations and model evaluation results that can be used to understand sales performance and customer behavior.

🎯 Objectives

Load and inspect the retail sales dataset.

Check missing values and duplicate records.

Validate numeric and categorical data.

Convert transaction dates into useful time-based features.

Calculate gross sales and discount amounts.

Analyze sales by:

Product category

Brand

Customer segment

Region

Sales channel

Payment method

Age group

Gender

Identify top-selling products and customers.

Study relationships between numerical variables using correlation analysis.

Build machine learning models to predict sales_amount.

Compare Linear Regression, Random Forest, and Gradient Boosting models.

Save visualizations, model results, and the trained prediction model.

📂 Project Structure

Retail-Sales-Analysis/
│
├── Retail Sales(1).ipynb
├── Data/
│   └── retail_sales_dataset.csv
│
├── Visualizations/
│   ├── monthly_sales_trend.png
│   ├── monthly_transactions.png
│   ├── sales_by_category.png
│   ├── quantity_by_category.png
│   ├── sales_by_brand.png
│   ├── sales_by_customer_segment.png
│   ├── sales_by_region.png
│   ├── sales_by_channel.png
│   ├── sales_by_payment_method.png
│   ├── sales_by_age_group.png
│   ├── sales_by_gender.png
│   ├── discount_vs_sales.png
│   ├── quantity_vs_sales.png
│   ├── top_10_products_by_sales.png
│   ├── top_10_products_by_quantity.png
│   ├── customer_segment_distribution.png
│   ├── average_sales_by_segment.png
│   ├── top_10_customers.png
│   ├── correlation_heatmap.png
│   ├── model_results.csv
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── prediction_error_distribution.png
│
└── retail_sales_prediction_model.pkl

Note: The exact output folder capitalization should match the paths used in the notebook. The notebook uses both visualizations and Visualizations in different cells, so keeping the folder naming consistent is recommended.

🛠️ Technologies & Libraries

Programming Language

Python 3.x

Libraries

Pandas – data loading, cleaning, transformation, and analysis

NumPy – numerical operations

Matplotlib – data visualization

Seaborn – correlation heatmap visualization

Scikit-learn – machine learning

Joblib – saving and loading the trained model

📊 Dataset

The notebook uses:

retail_sales_dataset.csv

The analysis works with fields including:

transaction_id

transaction_date

customer_id

product_id

product_name

category

brand

quantity

unit_price

discount_pct

sales_amount

customer_gender

customer_age_group

customer_segment

payment_method

sales_channel

region

🔍 Data Preparation & Quality Checks

The notebook performs several data-quality checks:

1. Dataset Inspection

Dataset shape

Column names

First five records

Data types

Descriptive statistics

2. Missing Values

The notebook calculates missing values for every column.

df.isnull().sum()

3. Duplicate Records

Duplicate rows and duplicate transaction IDs are checked.

df.duplicated().sum()
df["transaction_id"].duplicated().sum()

4. Date Conversion

The transaction date is converted to a Pandas datetime type.

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

5. Numeric Validation

The notebook checks for invalid values such as:

Quantity ≤ 0

Unit price ≤ 0

Discount below 0%

Discount above 100%

Sales amount ≤ 0

🧮 Feature Engineering

Several new features are created from the transaction date:

year
month
month_name
day
day_of_week
quarter

The notebook also calculates:

Gross Amount

gross_amount = quantity × unit_price

Discount Amount

discount_amount = gross_amount × (discount_pct / 100)

These additional features are used for sales analysis and correlation analysis.

📈 Exploratory Data Analysis

The notebook analyzes retail sales from multiple perspectives.

Sales Trend

Monthly sales trend

Monthly transaction volume

Product Analysis

Sales by category

Quantity by category

Sales by brand

Top 10 products by sales

Top 10 products by quantity

Customer Analysis

Sales by customer segment

Transaction distribution by customer segment

Average sales by customer segment

Top customers by transaction frequency

Top 10 customers by total sales

Sales by age group

Sales by gender

Business Analysis

Sales by region

Sales by sales channel

Sales by payment method

Average sales by discount percentage

Quantity vs. sales amount

📊 Visualizations

The project generates multiple charts using Matplotlib and Seaborn.

Examples include:

Monthly Sales Trend

Monthly Transaction Volume

Sales by Product Category

Sales by Brand

Sales by Region

Sales by Sales Channel

Sales by Payment Method

Top 10 Products

Top 10 Customers

Customer Segment Distribution

Correlation Heatmap

Model Comparison

Actual vs. Predicted Sales

Prediction Error Distribution

All generated visualization files are saved in the visualization output directory.

🤖 Sales Prediction

The project also builds machine learning models to predict:

Target = sales_amount

Features Used

Numerical Features

quantity
unit_price
discount_pct

Categorical Features

customer_gender
customer_age_group
customer_segment
category
brand
payment_method
sales_channel
region

Categorical variables are transformed using OneHotEncoder.

The data is split into:

80% Training Data
20% Testing Data

using:

train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

🧠 Machine Learning Models

The notebook evaluates three regression models:

1. Linear Regression

A baseline regression model used to predict sales amount from the selected features.

2. Random Forest Regressor

An ensemble model using multiple decision trees.

Configuration used in the notebook:

RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

3. Gradient Boosting Regressor

A boosting-based regression model used for comparison with Linear Regression and Random Forest.

📏 Model Evaluation

The models are evaluated using:

MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted sales.

RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

R² Score

Measures how much of the variation in the target variable is explained by the model.

The results are stored in:

model_results.csv

The notebook also creates a model comparison chart based on R² score.

📉 Prediction Analysis

For the Random Forest model, the notebook generates:

Actual vs Predicted Sales

Compares actual sales amounts with model predictions.

Prediction Error Distribution

Analyzes the distribution of:

prediction_error = actual_sales - predicted_sales

💾 Model Saving

The trained Random Forest pipeline is saved using Joblib:

joblib.dump(
    rf_model,
    "../retail_sales_prediction_model.pkl"
)

The saved model can later be loaded with:

loaded_model = joblib.load(
    "../retail_sales_prediction_model.pkl"
)

and used to generate predictions.

🚀 How to Run the Project

Step 1: Clone or Download the Project

Download the project files and place the dataset in the expected Data directory.

Step 2: Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

Step 3: Open the Notebook

jupyter notebook

Open:

Retail Sales(1).ipynb

Step 4: Update the Dataset Path

The notebook currently loads the CSV using a local Windows path:

pd.read_csv(
    "C:\\Thiranex Internship Tasks\\Retail Sales Analysis\\Data\\retail_sales_dataset.csv"
)

Change this path to match the location of the dataset on your computer.

For a project-relative structure, you can use:

df = pd.read_csv("Data/retail_sales_dataset.csv")

Step 5: Run All Cells

Run the notebook from top to bottom to:

Load the dataset.

Perform data-quality checks.

Engineer features.

Perform exploratory analysis.

Generate visualizations.

Train regression models.

Compare model performance.

Save the trained model and results.

📌 Key Outputs

The project produces:

Output

Purpose

monthly_sales_trend.png

Monthly sales trend

monthly_transactions.png

Monthly transaction volume

sales_by_category.png

Category-level sales

sales_by_brand.png

Brand-level sales

sales_by_region.png

Regional sales

sales_by_channel.png

Sales channel analysis

sales_by_payment_method.png

Payment method analysis

top_10_products_by_sales.png

Top products by sales

top_10_customers.png

Top customers by sales

correlation_heatmap.png

Numerical correlation analysis

model_results.csv

Model evaluation metrics

model_comparison.png

R² comparison

actual_vs_predicted.png

Prediction performance

prediction_error_distribution.png

Prediction error analysis

retail_sales_prediction_model.pkl

Saved Random Forest model

⚠️ Notes

The notebook assumes that the CSV dataset contains the columns referenced in the analysis.

The dataset path is currently machine-specific and should be updated before running on another computer.

The notebook contains two Random Forest training cells with the same model configuration; the second one repeats the training and prediction process.

The visualization directory naming should be made consistent because different cells reference visualizations and Visualizations.

The notebook uses sparse_output=False for the Gradient Boosting preprocessing pipeline, which requires a compatible version of scikit-learn.

👨‍💻 Project Skills Demonstrated

This project demonstrates practical skills in:

Python

Pandas

NumPy

Data Cleaning

Data Validation

Feature Engineering

Exploratory Data Analysis (EDA)

Data Visualization

Business Analytics

Customer Segmentation Analysis

Product Analysis

Statistical Correlation

Regression

Machine Learning Pipelines

One-Hot Encoding

Model Evaluation

Random Forest

Gradient Boosting

Model Serialization with Joblib

📄 Project Type

Retail Sales Analytics + Machine Learning Prediction Project

This project can be included in a Data Analyst, Business Analyst, or Junior Data Scientist portfolio to demonstrate an end-to-end workflow from raw retail transaction data to analysis, visualization, and predictive modeling.

👨‍💻 Author
Shubham Lad

🔗 GitHub
https://github.com/ladshubham742-bit/Shubham-Lad.

🔗 LinkedIn
www.linkedin.com/in/shubham-lad-314a66319.

⭐ If You Find This Project Useful
If you find this project helpful, consider giving the repository a ⭐ on GitHub!
