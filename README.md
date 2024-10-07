# Churn Prediction Analysis

## Project Overview
This project focuses on **predicting customer churn** using machine learning techniques. A **Random Forest Classifier** is used to model the problem, predicting whether a customer will stay or churn based on various features like service usage, customer demographics, and contract details.

### Objective
The goal is to help businesses identify customers who are likely to leave (churn) and take preventive measures to improve customer retention.

## Dataset
The dataset used contains customer information with features such as:
- **Demographic details** (e.g., Gender, State, Married status)
- **Service-related attributes** (e.g., Phone Service, Internet Service, Streaming Services)
- **Contract-related details** (e.g., Contract type, Payment method)

The target variable is `Customer_Status`, where:
- `0` = Stayed
- `1` = Churned

## Prerequisites
Ensure you have the following software and libraries installed before running the project:

1. **Python 3.x**
2. **Anaconda (recommended)**: [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)

### Python Libraries:
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **joblib**
- **openpyxl** (for handling Excel files)

You can install all required libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl joblib
```

## Setup Instructions
### 1. Clone or Download the Project
Download this project repository to your local machine.

### 2. Dataset Preparation
- Place your dataset file `Prediction_Data.xlsx` inside a directory that you can access from the code.
- Ensure the file has the following two sheets:
  - **vw_ChurnData**: Contains historical data for training and testing the model.
  - **vw_JoinData**: Contains new data for making predictions.

### 3. Running the Project
You can run this project in **Google Colab** or **Jupyter Notebook**.

#### Option 1: Running in Google Colab
1. **Upload the dataset** to your Google Drive.
2. Mount Google Drive in Colab using:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update file paths in the code to point to your dataset in Google Drive.
4. Run the notebook cell-by-cell to train the model, evaluate it, and make predictions.

#### Option 2: Running in Jupyter Notebook (Local Setup)
1. Open **Jupyter Notebook** from Anaconda or the terminal.
2. Update the `file_path` in the code to the path where your `Prediction_Data.xlsx` file is located.
3. Execute the notebook to train the model, evaluate it, and make predictions.

### 4. Model Training
The **Random Forest Classifier** is used to train on the historical data. Features are preprocessed, including encoding categorical variables and handling missing values (`NaN` values replaced with `0`).

### 5. Model Evaluation
After training, the model is evaluated using:
- **Confusion Matrix**: Shows how many predictions were correct or incorrect.
- **Classification Report**: Displays metrics like precision, recall, and F1-score for each class (Stayed/Churned).

### 6. Predicting New Data
Once the model is trained, it can be used to predict churn for new customers by loading data from the `vw_JoinData` sheet in the same Excel file.

## Key Sections
1. **Data Preprocessing**:
   - Dropping unnecessary columns like `Customer_ID`, `Churn_Category`, and `Churn_Reason`.
   - Encoding categorical variables using `LabelEncoder`.
   - Handling missing values (NaN values are replaced with 0).

2. **Model Training**:
   - **RandomForestClassifier** is used with default hyperparameters (`n_estimators=100` and `random_state=42`).
   
3. **Model Evaluation**:
   - The confusion matrix and classification report provide an overview of the model's performance.

4. **Prediction on New Data**:
   - Predicts churn status for new customers and saves the output to a CSV file (`Predictions.csv`).

## Results
- The trained Random Forest model can accurately predict customer churn with a certain level of confidence.
- The feature importance plot highlights the most relevant features in determining customer churn.

## File Structure
```
.
├── Churn_Prediction.ipynb    # Jupyter notebook containing the churn analysis code
├── Prediction_Data.xlsx      # Excel file with training and testing data (vw_ChurnData and vw_JoinData sheets)
└── Predictions.csv           # Output file with predicted churn results (generated after running the model)
```

## Usage
- Modify the dataset file paths as per your setup.
- Run the code block-by-block in Google Colab or Jupyter Notebook.
- Once trained, use the model to predict churn on new customer data.
