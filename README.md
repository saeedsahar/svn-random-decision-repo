- **Name**: Saher Saeed  
- **Student ID**: 23095056  
- **GitHub Repository**: [Model-Comparison-Repo](https://github.com/saeedsahar/svn-random-decision-repo.git)


# Apartment for Rent Classified

This notebook analyzes the **Apartment for Rent Classified** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/apartment+for+rent+classified).

- **Source**: UCI Machine Learning Repository  
- **Data type**: Tabular data containing apartment rental listings and attributes  
- **Features**: Include number of bedrooms, bathrooms, square footage, furnishing status, amenities (elevator, parking, security), and location info  
- **Target**: Classification of apartment listings, possibly based on rent level or type  
- **Purpose**: To predict apartment rental categories based on listing attributes and analyze key factors influencing rental classification

---

## Data Card

| Property               | Details                                         |
|------------------------|--------------------------------------------------|
| **Dataset Name**       | Apartment for Rent Classified                   |
| **UCI ID**             | 555                                             |
| **Total Rows**         | 99826       |
| **Total Columns**      | 21        |
| **Data Type**          | Tabular (Structured)                            |
| **Target Variable**    | Rental Category/Class                           |
| **Missing Values**     | To be checked                                   |
| **Source**             | [UCI Repository Link](https://archive.ics.uci.edu/ml/datasets/apartment+for+rent+classified) |

---

## Variable Overview

The table below lists all features (input variables) in the dataset, along with their data types and descriptions (if provided).

| Column Name       | Data Type | Description                       |
|-------------------|-----------|-----------------------------------|
| `bedrooms`        | Numeric   | Number of bedrooms                |
| `bathrooms`       | Numeric   | Number of bathrooms               |
| `square_feet`     | Numeric   | Area in square feet               |
| `furnishing`      | Categorical | Furnishing status (furnished/unfurnished) |
| `location`        | Categorical | Location or region                |
| `elevator`        | Binary    | Presence of elevator (0/1)        |
| `parking`         | Binary    | Availability of parking (0/1)     |
| `security`        | Binary    | Presence of security (0/1)        |
| `latitude`        | Numeric   | GPS Latitude                      |
| `longitude`       | Numeric   | GPS Longitude                     |
| `price`           | Numeric   | Rental price                      |
| `time`            | Numeric   | Timestamp or time-related metric  |
| `target`          | Categorical | Rental category/class label      |

> *Note: Update column descriptions based on `apartment_for_rent_classified.variables` if you want full accuracy.*

---

**Citation**:  
> Tareq Nasir and Saeed Aljahdali. An Intelligent System for Predicting Apartment for Rent Category Using Machine Learning Algorithms. *International Journal of Advanced Computer Science and Applications*, Vol. 13, No. 8, 2022.  
> DOI: 10.14569/IJACSA.2022.0130882


##  Load Dataset from UCI Repository

We use the `ucimlrepo` Python package to fetch the **Apartment for Rent Classified** dataset directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/apartment+for+rent+classified).




##  Data Cleaning: Handling Duplicates, Missing Values, Type Conversion and Outliers


##  Outlier Detection: Before vs After

The box plots below show a side-by-side comparison of numerical features **before and after outlier removal**.

- **Left plot**: Original data with potential outliers.
- **Right plot**: Cleaned data after removing extreme values.

Using soothing pastel colors, these visualizations help us clearly see how outlier treatment affects the distribution of each feature. This step ensures the data is more consistent and less skewed, improving model performance and interpretability.




##  Correlation Matrix of Numerical Features

The heatmap below shows the **Pearson correlation coefficients** between all numerical features in the dataset.

- Values range from **-1 to +1**:
  - `+1`: Perfect positive correlation
  - `-1`: Perfect negative correlation
  - `0`: No linear correlation

### Key Observations:
- `square_feet` is highly correlated with both `bathrooms` (0.78) and `bedrooms` (0.72), suggesting that larger apartments typically have more rooms.
- `price` has moderate correlations with `square_feet` (0.39), `bathrooms`, and `bedrooms` (both 0.32).
- `latitude`, `longitude`, and `time` show minimal correlation with most other features, indicating spatial and temporal independence in pricing or size.

This matrix helps identify multicollinearity and informs feature selection for modeling.


##  Distribution of Numerical Features

The histograms below display the distribution of each numerical feature in the dataset. These visualizations help identify data skewness, modality, and potential outliers.

### Key Insights:
- **price** and **square_feet** show strong right-skewness, indicating most listings fall on the lower end with a few high-value outliers.
- **bedrooms** appears to have a multimodal distribution, reflecting common apartment configurations (e.g., 1BHK, 2BHK, etc.).
- **latitude** and **longitude** exhibit distinct location-based clusters, likely corresponding to different regions or cities.
- **time** shows a spike at the end, possibly indicating a batch of recent listings or timestamp artifacts.

These distributions guide data transformation steps like normalization or log-scaling for improved model performance.


##  Average Rent Price by State

The bar chart below shows the **average rental price** across U.S. states, with **listing counts labeled on top of each bar**.

### Key Insights:
- **Hawaii (HI)** and **California (CA)** have the highest average rents, reflecting the premium cost of living in these states.
- **New York (NY)**, **Massachusetts (MA)**, and **Washington D.C.** also rank high in average rent.
- States like **Mississippi (MS)**, **North Dakota (ND)**, and **Wyoming (WY)** have the **lowest average rents**, often paired with fewer listings.
- The number on each bar represents the **number of listings** considered for that state, providing context on data reliability (e.g., high rent in HI is based on just 3 listings).
- * A red asterisk above the bar indicates the state has fewer than 5 listings, and therefore its average rent value may not be statistically reliable.

This chart helps identify regional pricing patterns and supports geographic segmentation for rental market analysis.


## Daily Average Rent Price Over Time

The line plot below illustrates the **daily average rent price** throughout the year **2019**.

### Key Observations:
- The rent prices show **high variability** across days, with several noticeable peaks and drops.
- A sharp spike around **mid-2019** indicates either an outlier listing or a surge in high-priced rentals on that specific day.
- There are periods of relative stability, especially toward the end of the year, where fluctuations decrease.

This visualization helps reveal **temporal patterns** in rental pricing and can inform decisions related to **seasonality**, **anomaly detection**, or **trend analysis**.


##  Model Performance Comparison

Three machine learning models were trained and evaluated to classify apartment rent categories:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**

###  Summary:

- **SVM** showed limited performance, indicating underfitting.
- **Random Forest** performed the best, offering strong generalization and balanced classification.
- **Decision Tree** performed well but showed signs of overfitting.

Among the three, **Random Forest** proved to be the most effective for this classification task.



##  Hyperparameter Tuning with GridSearchCV

To improve model performance, we applied **GridSearchCV** for systematic hyperparameter tuning across three classifiers:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**

Each model was paired with a relevant set of hyperparameters:
- **SVM**: Regularization strength (`C`) and kernel type
- **Random Forest**: Number of trees and maximum depth
- **Decision Tree**: Tree depth and minimum samples per split

###  Outcome:
Grid search was performed using **3-fold cross-validation** and **accuracy** as the scoring metric. The best parameters for each model were stored for final evaluation.

This ensures that each classifier operates under optimized settings for the dataset at hand.



## Hyperparameter Tuning Results (Grid Search Visualization)

The plots below illustrate how different hyperparameter combinations affect **model accuracy** during 3-fold cross-validation for each classifier:

- **SVM**: Accuracy is plotted against the regularization parameter `C`, with separate lines for each `kernel` type.
- **Random Forest**: Shows how the number of trees (`n_estimators`) and maximum depth (`max_depth`) impact performance.
- **Decision Tree**: Displays accuracy changes based on `max_depth` and `min_samples_split`.

### Insights:
- These tuning curves help identify the **optimal hyperparameter settings** for each model.
- Visual trends reveal how **complexity controls** (like tree depth or regularization strength) affect model generalization.
- Models with too low or too high complexity often perform worse, reaffirming the importance of balanced tuning.

Such visual diagnostics guide effective model selection and fine-tuning for improved performance on unseen data.


##  Model Evaluation (After Hyperparameter Tuning)

Each tuned model was evaluated on the validation set to assess its final performance:

- **Predictions** were generated using the best estimator from `GridSearchCV`.
- **Accuracy** and detailed **classification reports** were computed for comparison.
- **Confusion matrices** were plotted to visualize true vs. predicted labels.

### Key Insights:
- Accuracy gives a quick snapshot of performance, but precision, recall, and F1-score offer a deeper view—especially in imbalanced scenarios.
- The confusion matrices help identify where each model tends to misclassify, guiding future improvements (e.g., handling false positives/negatives).

This step validates how well the tuned models generalize to unseen data.


## Accuracy Comparison: Before vs After Tuning

The bar chart below compares the **validation accuracy** of each model **before and after hyperparameter tuning**.

### Key Observations:
- All models show a noticeable improvement after tuning, confirming the benefit of optimized hyperparameters.
- **Random Forest** achieved the highest gain, demonstrating strong generalization with fine-tuned parameters.
- **SVM** also improved significantly, though it still trails behind ensemble models.
- **Decision Tree** saw modest improvement, likely due to its tendency to overfit without pruning.

This visual clearly highlights the impact of tuning on model performance and supports the importance of parameter optimization in machine learning workflows.


## Confusion Matrix Comparison: Train vs Test

The heatmaps below visualize the **confusion matrices** for each model on both the **training** and **test** datasets.

Each row represents a model (SVM, Random Forest, Decision Tree), and the columns show:
- **Left**: Training set performance
- **Right**: Test set performance

### Key Insights:
- These visualizations help assess how well each model generalizes from training to test data.
- A strong diagonal indicates high classification accuracy.
- Differences between train and test matrices reveal **overfitting** or **underfitting** tendencies.
  - e.g., a model with perfect training results but poor test performance may be overfitting.

This comparison is essential for validating the robustness and generalizability of each classifier.


##  Final Model Evaluation Summary

The table below summarizes the performance of the **tuned models** on the validation set, including:

- **Accuracy**: Overall correctness of predictions  
- **Precision**: Ability to avoid false positives (weighted)  
- **Recall**: Ability to identify true positives (weighted)  
- **F1 Score**: Harmonic mean of precision and recall (weighted)  
- **Prediction Time**: Time taken to generate predictions on the validation set

This summary offers a side-by-side comparison of models in terms of both **predictive performance** and **efficiency**, helping select the most balanced option for deployment.


## Model Comparison: Accuracy vs. Training Time

The plot below compares three classifiers — **SVM**, **Decision Tree**, and **Random Forest** — based on their **accuracy scores** and **training time**.

### Plot Details:
- The **bar chart** (blue) shows the **accuracy score** of each model on the test set.
- The **line plot** (red) shows the **fit time (in seconds)** on a secondary y-axis.

### Key Insights:
- This dual-axis plot highlights the trade-off between **performance** and **computational cost**.
- **Random Forest** may achieve higher accuracy but with longer training time.
- **Decision Tree** typically trains fastest, while **SVM** strikes a balance between time and accuracy.

This visualization helps in selecting the most appropriate model based on both **effectiveness** and **efficiency**, depending on project constraints.


## Example Code

```python

# === Data & Utilities ===
import pandas as pd
import numpy as np
import zipfile, requests, io
from ucimlrepo import fetch_ucirepo
from tabulate import tabulate
import time

# === Statistics & Distributions ===
from scipy.stats import kurtosis, skew

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# === Scikit-learn Core ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# === Models ===
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# === Evaluation Metrics ===
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

```
