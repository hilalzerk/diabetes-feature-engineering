#############################################
# DIABETES FEATURE ENGINEERING PROJECT
#############################################


# Business Problem

# The goal is to develop a machine learning model that can predict
# whether a person has diabetes based on given features.
# Before building the model, the required data analysis and
# feature engineering steps must be completed.


# The dataset is part of a large database maintained by the
# National Institute of Diabetes and Digestive and Kidney Diseases in the USA.
# It contains data from a diabetes study conducted on Pima Indian women
# aged 21 and older living in Phoenix, Arizona — the 5th largest city in Arizona.
# The target variable "Outcome" indicates whether the diabetes test result
# is positive (1) or negative (0).

# =============================================================================
# DATASET COLUMN DESCRIPTIONS
# =============================================================================

# | Column Name              | Description
# |--------------------------|-----------------------------------------------------------------------
# | Pregnancies              | Number of pregnancies
# | Glucose                  | 2-hour plasma glucose concentration in oral glucose tolerance test
# | BloodPressure            | Diastolic blood pressure (mm Hg)
# | SkinThickness            | Triceps skin fold thickness (mm)
# | Insulin                  | 2-hour serum insulin (mu U/ml)
# | DiabetesPedigreeFunction | A function that scores the likelihood of diabetes based on family history
# | BMI                      | Body Mass Index (weight in kg / height in m^2)
# | Age                      | Age (years)
# | Outcome                  | Has the disease (1) or not (0)
# =============================================================================

# Task 1 : Exploratory Data Analysis
# Step 1: Examine the general picture.
# Step 2: Capture numeric and categorical variables.
# Step 3: Analyze numeric and categorical variables.
# Step 4: Perform target variable analysis.
#         (Mean of target by categorical variables, mean of numeric variables by target)
# Step 5: Perform outlier analysis.
# Step 6: Perform missing value analysis.
# Step 7: Perform correlation analysis.
# =============================================================================
# Task 2 : Feature Engineering
# Step 1: Handle missing and outlier values.
#         Although there are no missing observations in the dataset,
#         variables like Glucose and Insulin may contain 0 values that represent missing data.
#         For example, a person's glucose or insulin value cannot be 0.
#         Replace zero values with NaN and apply missing value procedures.
# Step 2: Create new variables.
# Step 3: Perform encoding operations.
# Step 4: Standardize numeric variables.
# Step 5: Build a model.
# =============================================================================

# Required libraries and settings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Output directory for saving plots
SAVE_DIR = "/Users/hilalzerkdemirkan/PycharmProjects/Miuul- Bootcamp/Ödevler/Diabets/Grafikler"
os.makedirs(SAVE_DIR, exist_ok=True)


# Load dataset
def load():
    data = pd.read_csv("Ödevler/Diabets/Data/diabetes.csv")
    return data

dataframe = load()

# =============================================================================
# Task 1 : Exploratory Data Analysis
# =============================================================================
# Step 1: Examine the general picture.

def check_df(dataframe, head=5):
    print("########## Shape ########")
    print(dataframe.shape)

    print("########## Types ########")
    print(dataframe.dtypes)

    print("########## Head ########")
    print(dataframe.head(head))

    print("########## Tail ########")
    print(dataframe.tail(head))

    print("########## NA ########")
    print(dataframe.isnull().sum())

    print("########## Quantiles ########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(dataframe)

# Step 2: Capture numeric and categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numeric, and cardinal variables in the dataset.
    Note: Numeric-looking categorical variables are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be extracted
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optional
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numeric variables
        cat_but_car: list
                List of cardinal-looking categorical variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is included in cat_cols.
        The sum of the 3 returned lists equals the total number of variables.
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

# Step 3: Analyze numeric and categorical variables.

# Categorical Variable Analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette="viridis")
        plt.show(block=False)

for col in cat_cols:
     cat_summary(dataframe, col)


# Numeric Variable Analysis
def num_summary(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    dataframe[num_cols].describe().T
    print(dataframe[num_cols].describe().T)

    if plot:
        dataframe[num_cols].hist()
        plt.xlabel(num_cols)
        plt.ylabel(num_cols)
        plt.show(block=True)

for col in num_cols:
    num_summary(dataframe, col, plot=False)


# Step 4: Target variable analysis.
# (Mean of target by categorical variables, mean of numeric variables by target)

# Target analysis by numeric variable
def target_summary_with_num(dataframe, target, numerical_col):
    temp_df = dataframe.groupby(target).agg({numerical_col: "mean"})
    print(temp_df)
    print("##################################################")
    colors = ["#cb4854", "#e49e7f"]
    temp_df.plot(kind="bar", y=numerical_col, color=colors)
    plt.title(f"Mean of {numerical_col} by {target}")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/target_vs_{numerical_col}.png", dpi=150)
    plt.close()
    print(f"  ✓ target_vs_{numerical_col}.png saved")

for col in num_cols:
    target_summary_with_num(dataframe, "Outcome", col)


# Step 5: Outlier analysis.

# Calculate lower and upper threshold values using IQR method
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Check whether there are outliers - returns True/False
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Access the outlier values themselves
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


print("IS THERE AN OUTLIER FOR EACH COLUMN?")
for col in num_cols:
    low, up = outlier_thresholds(dataframe, col)
    print(f"  {col:30s} | Outlier: {check_outlier(dataframe, col)} | Low: {low:.2f} | Up: {up:.2f}")


# Step 6: Missing value analysis.

# Although there are no visible missing values, some columns cannot have 0 values.
zero_columns = [col for col in dataframe.columns if (dataframe[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

zero_columns


# Step 7: Correlation analysis.

# Correlation analysis
dataframe.corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix Before Feature Engineering", fontsize=20)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/correlation_matrix_before_fe.png", dpi=150)
plt.close()
print("✓ correlation_matrix_before_fe.png saved")

# Base Model Setup

y = dataframe["Outcome"]  # dependent variable - target
X = dataframe.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Feature importance table
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num], palette="viridis")
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/feature_importance_base_model.png", dpi=150)
    plt.close()
    print("✓ feature_importance_base_model.png saved")
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)


# =============================================================================
# Task 2 : Feature Engineering
# =============================================================================

# Step 1: Handle missing and outlier values.
# Although there are no missing observations, variables like Glucose and Insulin
# may contain 0 values that represent missing data.
# A person's glucose or insulin value cannot be 0.
# Replace zero values with NaN and then apply missing value procedures.

# Assign NaN values
for col in zero_columns:
    dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])

# Missing value analysis after NaN assignment
dataframe.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(dataframe, na_name=True)


### FILLING MISSING VALUES
# Improvement: Fill by Outcome group median
# Diabetic patients have different insulin/glucose values → group-based filling is more meaningful

for col in zero_columns:
    dataframe[col] = dataframe.groupby("Outcome")[col].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"  {col:20s} → Filled with Median by Outcome group")

# Fill any remaining missing values with global median
for col in zero_columns:
    dataframe[col] = dataframe[col].fillna(dataframe[col].median())

print("\n>> Missing value check after filling:")
print(dataframe.isnull().sum())


# Outlier Suppression

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))
    if check_outlier(dataframe, col):
        replace_with_thresholds(dataframe, col)

for col in dataframe.columns:
    print(col, check_outlier(dataframe, col))


# Step 2: Create new variables.

# --- Age Category ---
dataframe.loc[(dataframe["Age"] >= 21) & (dataframe["Age"] < 50), "NEW_AGE_CAT"] = "mature"
dataframe.loc[(dataframe["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# --- BMI Category ---
# World Health Organization classification
dataframe.loc[(dataframe["BMI"] < 18.5), "NEW_BMI"] = "Underweight"
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25), "NEW_BMI"] = "Healthy"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30), "NEW_BMI"] = "Overweight"
dataframe.loc[(dataframe["BMI"] >= 30), "NEW_BMI"] = "Obese"

# --- Glucose Category ---
dataframe.loc[(dataframe["Glucose"] < 70), "NEW_GLUCOSE"] = "Low"
dataframe.loc[(dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100), "NEW_GLUCOSE"] = "Normal"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126), "NEW_GLUCOSE"] = "Prediabetes"
dataframe.loc[(dataframe["Glucose"] >= 126), "NEW_GLUCOSE"] = "Diabetes"

# --- Age + BMI Interaction ---
dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "underweightmature"
dataframe.loc[(dataframe["BMI"] < 18.5) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
dataframe.loc[
    (dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "healthymature"
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25) & (
            dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30) & (
            dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "overweightmature"
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30) & (
            dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
dataframe.loc[(dataframe["BMI"] >= 30) & (dataframe["Age"] < 50), "NEW_AGE_BMI_NOM"] = "obesemature"
dataframe.loc[(dataframe["BMI"] >= 30) & (dataframe["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# --- Age + Glucose Interaction ---
dataframe.loc[(dataframe["Glucose"] < 100) & (dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
dataframe.loc[(dataframe["Glucose"] < 100) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126) & (
            dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "prediabetesmature"
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126) & (
            dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "prediabetessenior"
dataframe.loc[(dataframe["Glucose"] >= 126) & (dataframe["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
dataframe.loc[(dataframe["Glucose"] >= 126) & (dataframe["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


# --- Insulin Category ---
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

dataframe["NEW_INSULIN_SCORE"] = dataframe.apply(set_insulin, axis=1)

# --- Glucose * Insulin Interaction ---
dataframe["NEW_GLUCOSE_INSULIN"] = dataframe["Glucose"] * dataframe["Insulin"]

# --- Age * BMI Numeric Interaction ---
dataframe["NEW_AGE_BMI"] = dataframe["Age"] * dataframe["BMI"]

# Glucose / Insulin ratio (insulin resistance indicator)
dataframe["NEW_GLUCOSE_INSULIN_RATIO"] = dataframe["Glucose"] / (dataframe["Insulin"] + 1)

# BMI * Pedigree (genetic + obesity risk)
dataframe["NEW_BMI_PEDIGREE"] = dataframe["BMI"] * dataframe["DiabetesPedigreeFunction"]

# Blood Pressure * Age (increasing BP risk with age)
dataframe["NEW_BP_AGE"] = dataframe["BloodPressure"] * dataframe["Age"]

# Pregnancy / Age ratio
dataframe["NEW_PREG_AGE"] = dataframe["Pregnancies"] / dataframe["Age"]

# Log transformations (for skewed distributions)
dataframe["NEW_INSULIN_LOG"] = np.log1p(dataframe["Insulin"])
dataframe["NEW_PEDIGREE_LOG"] = np.log1p(dataframe["DiabetesPedigreeFunction"])
dataframe["NEW_AGE_LOG"] = np.log1p(dataframe["Age"])
dataframe["NEW_PREG_LOG"] = np.log1p(dataframe["Pregnancies"])

# High-risk combination flags
dataframe["NEW_HIGH_RISK"] = ((dataframe["Glucose"] >= 126) & (dataframe["BMI"] >= 30)).astype(int)
dataframe["NEW_ELDERLY_OBESE"] = ((dataframe["Age"] >= 50) & (dataframe["BMI"] >= 30)).astype(int)
dataframe["NEW_HIGH_GLUCOSE_INSULIN"] = ((dataframe["Glucose"] >= 126) & (dataframe["Insulin"] > 166)).astype(int)

# Ordinal categories (numerically encoded)
dataframe.loc[dataframe["Glucose"] < 70, "NEW_GLUCOSE_CAT"] = 0
dataframe.loc[(dataframe["Glucose"] >= 70) & (dataframe["Glucose"] < 100), "NEW_GLUCOSE_CAT"] = 1
dataframe.loc[(dataframe["Glucose"] >= 100) & (dataframe["Glucose"] < 126), "NEW_GLUCOSE_CAT"] = 2
dataframe.loc[dataframe["Glucose"] >= 126, "NEW_GLUCOSE_CAT"] = 3

dataframe.loc[dataframe["BMI"] < 18.5, "NEW_BMI_CAT"] = 0
dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25), "NEW_BMI_CAT"] = 1
dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30), "NEW_BMI_CAT"] = 2
dataframe.loc[dataframe["BMI"] >= 30, "NEW_BMI_CAT"] = 3

dataframe.loc[dataframe["Pregnancies"] == 0, "NEW_PREG_CAT"] = 0
dataframe.loc[(dataframe["Pregnancies"] >= 1) & (dataframe["Pregnancies"] <= 3), "NEW_PREG_CAT"] = 1
dataframe.loc[(dataframe["Pregnancies"] >= 4) & (dataframe["Pregnancies"] <= 7), "NEW_PREG_CAT"] = 2
dataframe.loc[dataframe["Pregnancies"] > 7, "NEW_PREG_CAT"] = 3

dataframe.loc[dataframe["DiabetesPedigreeFunction"] < 0.3, "NEW_PEDIGREE_CAT"] = 0
dataframe.loc[(dataframe["DiabetesPedigreeFunction"] >= 0.3) &
              (dataframe["DiabetesPedigreeFunction"] < 0.6), "NEW_PEDIGREE_CAT"] = 1
dataframe.loc[dataframe["DiabetesPedigreeFunction"] >= 0.6, "NEW_PEDIGREE_CAT"] = 2

dataframe.loc[dataframe["BloodPressure"] < 60, "NEW_BP_CAT"] = 0
dataframe.loc[(dataframe["BloodPressure"] >= 60) &
              (dataframe["BloodPressure"] < 80), "NEW_BP_CAT"] = 1
dataframe.loc[dataframe["BloodPressure"] >= 80, "NEW_BP_CAT"] = 2

print("\n>> New variables created:")
new_cols = [col for col in dataframe.columns if col.startswith("NEW_")]
for col in new_cols:
    print(f"  {col}")

print(f"\n>> New shape: {dataframe.shape}")

print("\n>> Relationship between new variables and Outcome:")
for col in new_cols:
    if dataframe[col].dtype == "O":
        print(f"\n--- {col} ---")
        print(dataframe.groupby(col).agg({"Outcome": ["mean", "count"]}))


# Step 3: Encoding operations.

# Label Encoding (binary categorical variables with 2 classes)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() == 2]

for col in binary_cols:
    dataframe = label_encoder(dataframe, col)

dataframe.head()

# One-Hot Encoding (categorical variables with more than 2 classes)
cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

dataframe.head()

# Step 4: Standardize numeric variables.

cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
num_cols = [col for col in num_cols if col not in ["Outcome"]]

scaler = RobustScaler()
dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

dataframe.head()

# =============================================================================
# Correlation Matrix — After Feature Engineering
# =============================================================================

# Select only numeric columns (encoded columns included)
numeric_df = dataframe.select_dtypes(include=[np.number])

f, ax = plt.subplots(figsize=[22, 18])
sns.heatmap(numeric_df.corr(), annot=False, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix After Feature Engineering", fontsize=20)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/correlation_matrix_after_fe.png", dpi=150)
plt.close()
print("✓ correlation_matrix_after_fe.png saved")


# Step 5: Build the model.

# =============================================================================
# Random Forest Model
# =============================================================================
y = dataframe["Outcome"]
X = dataframe.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 3)}")
print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred), 2)}")
print(f"Auc: {round(roc_auc_score(y_test, y_pred), 2)}")

# Feature Importance — Random Forest
feature_imp_rf = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(feature_imp_rf.to_string())

plt.figure(figsize=(10, 10))
sns.barplot(x="Importance", y="Feature",
            data=feature_imp_rf.head(20), palette="viridis")
plt.title("Random Forest — Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_importance_rf.png", dpi=150)
plt.close()
print("✓ feature_importance_rf.png saved")

# Confusion Matrix — Random Forest
cm_rf = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm_rf, display_labels=["Healthy (0)", "Diabetes (1)"]).plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix — Random Forest", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/confusion_matrix_rf.png", dpi=150)
plt.close()
print("✓ confusion_matrix_rf.png saved")

# =============================================================================
# XGBoost Model
# =============================================================================
xgb_model = XGBClassifier(random_state=46, eval_metric="logloss").fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Comparison Table
print("=" * 55)
print(f"{'Metric':<12} {'Random Forest':>18} {'XGBoost':>18}")
print("=" * 55)
print(f"{'Accuracy':<12} {round(accuracy_score(y_test, y_pred), 3):>18.3f} {round(accuracy_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'Recall':<12} {round(recall_score(y_test, y_pred), 3):>18.3f} {round(recall_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'Precision':<12} {round(precision_score(y_test, y_pred), 3):>18.3f} {round(precision_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'F1':<12} {round(f1_score(y_test, y_pred), 3):>18.3f} {round(f1_score(y_test, y_pred_xgb), 3):>18.3f}")
print(f"{'AUC':<12} {round(roc_auc_score(y_test, y_pred), 3):>18.3f} {round(roc_auc_score(y_test, y_pred_xgb), 3):>18.3f}")
print("=" * 55)

# Feature Importance — XGBoost
feature_imp_xgb = pd.DataFrame({
    "Feature": X_train.columns,
    "XGBoost_Importance": xgb_model.feature_importances_
}).sort_values("XGBoost_Importance", ascending=False)

print("\nXGBoost Top 10 Features:")
print(feature_imp_xgb.head(10).to_string(index=False))

plt.figure(figsize=(10, 10))
sns.barplot(x="XGBoost_Importance", y="Feature",
            data=feature_imp_xgb.head(20), palette="viridis")
plt.title("XGBoost — Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/feature_importance_xgb.png", dpi=150)
plt.close()
print("✓ feature_importance_xgb.png saved")

# Confusion Matrix — XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm_xgb, display_labels=["Healthy (0)", "Diabetes (1)"]).plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix — XGBoost", fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/confusion_matrix_xgb.png", dpi=150)
plt.close()
print("✓ confusion_matrix_xgb.png saved")

# =============================================================================
# Random Forest vs XGBoost Comparison Chart
# =============================================================================
metrics = ["Accuracy", "Recall", "Precision", "F1", "AUC"]
rf_scores = [
    round(accuracy_score(y_test, y_pred), 3),
    round(recall_score(y_test, y_pred), 3),
    round(precision_score(y_test, y_pred), 3),
    round(f1_score(y_test, y_pred), 3),
    round(roc_auc_score(y_test, y_pred), 3)
]
xgb_scores = [
    round(accuracy_score(y_test, y_pred_xgb), 3),
    round(recall_score(y_test, y_pred_xgb), 3),
    round(precision_score(y_test, y_pred_xgb), 3),
    round(f1_score(y_test, y_pred_xgb), 3),
    round(roc_auc_score(y_test, y_pred_xgb), 3)
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, rf_scores, width, label="Random Forest", color="#2ecc71")
bars2 = ax.bar(x + width/2, xgb_scores, width, label="XGBoost", color="#3498db")

ax.set_ylim(0.75, 0.95)
ax.set_ylabel("Score")
ax.set_title("Random Forest vs XGBoost — Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/model_comparison_rf_vs_xgb.png", dpi=150)
plt.close()
print("✓ model_comparison_rf_vs_xgb.png saved")

print(f"\n✅ All plots saved to: {SAVE_DIR}")
