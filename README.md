# Student Performance Prediction (Math Score)

This project builds an end-to-end machine learning pipeline to predict a student’s **math score** based on demographics, parental background, lunch type, test preparation, and other test scores.  

It includes:

- **Exploratory Data Analysis (EDA)** in notebooks.
- A **modular training pipeline** (`src/`) for ingestion, transformation, and model training.
- A **Flask web app** (`application.py`) for interactive prediction.

The best-performing model in this project is:

- **Model:** Linear Regression  
- **Test R²:** **0.8804**

---

## Project Overview

### Objective

Predict a student’s math score given:

- Gender  
- Race/ethnicity  
- Parental level of education  
- Lunch type  
- Test preparation course  
- Reading score  
- Writing score  

This simulates real-world use cases in education analytics, risk modeling, and resource allocation where decision makers need accurate predictions of performance given observable features.

---

## Data Architecture & Project Structure

This project follows a simple but production-oriented data and code architecture.

### Data flow

1. **Raw data**  
   - Source: `StudentsPerformance.csv` (1,000 rows, 8 columns).  
   - Located under the project (e.g. `notebook/StudentsPerformance.csv` for the pipeline).

2. **Data ingestion (`DataIngestion`)**
   - Reads the raw CSV.
   - Splits into **train** and **test** sets (80/20).
   - Saves:
     - `artifact/raw.csv`
     - `artifact/train.csv`
     - `artifact/test.csv`

3. **Data transformation (`DataTransformation`)**
   - Builds a `ColumnTransformer` with:
     - **Numeric features:** `reading score`, `writing score`
       - Median imputation + `StandardScaler`.
     - **Categorical features:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`
       - Most-frequent imputation + `OneHotEncoder` + `StandardScaler(with_mean=False)`.
   - Fits on training data, transforms train & test.
   - Saves fitted preprocessor to:
     - `artifact/preprocessor.pkl`

4. **Model training (`ModelTrainer`)**
   - Uses transformed arrays (features + target) from the transformation step.
   - Trains and evaluates multiple regressors with hyperparameter tuning (via `GridSearchCV`):
     - Linear Regression
     - DecisionTreeRegressor
     - RandomForestRegressor
     - GradientBoostingRegressor
     - XGBRegressor
     - CatBoostRegressor
     - AdaBoostRegressor
   - Evaluation metric: **R² on test set**.
   - Selects the best model (by test R²), enforces a minimum performance threshold.
   - Saves the best model to:
     - `artifact/model.pkl`

5. **Prediction pipeline (`PredictPipeline`)**
   - Loads:
     - `artifact/preprocessor.pkl`
     - `artifact/model.pkl`
   - Applies the preprocessor and runs the model to produce predictions from new input data.

6. **Web application (`application.py`)**
   - Flask app with:
     - `/` – index page
     - `/predictdata` – form to collect user inputs and return predicted math score
   - Uses `CustomData` (in `predict_pipeline.py`) to convert form inputs into a pandas DataFrame consistent with training features.

---
## Modeling Approach

### Target

- **Target variable:** `math score`

### Features

- **Numeric**
  - `reading score`
  - `writing score`

- **Categorical**
  - `gender`
  - `race/ethnicity`
  - `parental level of education`
  - `lunch`
  - `test preparation course`

### Preprocessing

All preprocessing is implemented using **pipeline-based transformations** in scikit-learn:

- Imputation for missing values:
  - **Numeric:** median
  - **Categorical:** most frequent
- One-hot encoding for categorical variables
- Standardization for:
  - Numeric features
  - One-hot encoded features (using `StandardScaler(with_mean=False)`)

### Candidate Models

All models are trained and evaluated on a **common preprocessed feature space** and, in the production pipeline, tuned using `GridSearchCV`:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- CatBoost Regressor  
- AdaBoost Regressor  

### Metric

- **Primary evaluation metric:** R² score
- **Validation strategy:**  
  - 80/20 train–test split  
  - R² computed on the held-out **20% test set**

### Final Result

- **Best model:** Linear Regression  
- **Test R²:** **0.8804**

This indicates that the linear model explains approximately **88% of the variance** in math scores on unseen data, given the selected set of features and preprocessing steps.

---

## Repository Structure

Core layout (Python project folder):

```text
.
├── application.py                 # Flask application entrypoint
├── setup.py                       # Packaging configuration
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Ignored files and folders
├── .ebextensions/
│   └── python.config              # Elastic Beanstalk WSGI configuration
├── src/
│   ├── __init__.py
│   ├── exception.py               # CustomException and error formatting
│   ├── logger.py                  # Global logging configuration
│   ├── utils.py                   # save/load objects, model evaluation
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # DataIngestion & config
│   │   ├── data_transformation.py # DataTransformation & config
│   │   └── model_trainer.py       # ModelTrainer & config
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py      # Orchestrates training pipeline
│       └── predict_pipeline.py    # Prediction pipeline & CustomData
├── templates/
│   ├── index.html                 # Landing page
│   └── home.html                  # Input form + prediction display
└── notebook/
    ├── 1 . EDA STUDENT PERFORMANCE .ipynb   # EDA notebook
    └── 2. MODEL TRAINING.ipynb              # Model training notebook


