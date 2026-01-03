# Modeling: Raw vs PCA vs PLS with Logistic Regression, Decision Tree, and SVM

## Overview

This repository compares three classifiers:

• Logistic Regression
• Decision Tree
• Support Vector Machine (SVM)

Each model is evaluated on three feature spaces:

• Raw features (data/features.csv)
• PCA-decomposed features (analysis/pca/pca_space.csv)
• PLS-decomposed features (analysis/pls/pls_space.csv)

Model comparison is done using LOOCV (leave-one-out cross validation).
Once the best model + feature space is identified, a final model is fit on all data and used to generate prediction maps.

## Environment Setup
Activate your conda environment:

This repository was developed and tested using conda / mamba.
If you do not already have conda, install Miniforge (recommended for macOS / ARM):

https://github.com/conda-forge/miniforge

After installation, open a new terminal.

### Step 0: Clone the Repository

git clone <REPO_URL>
cd modeling

### Step 1: Create the Conda Environment

Create a fresh environment called geo_env (or whatever you want to call it):

`conda create -n geo_env python=3.9 -y`

Activate it:

`conda activate geo_env`

### Step 2: Install Dependencies

All required Python packages are listed in requirements.txt.

Install them using pip:

`pip install --upgrade pip`
`pip install -r requirements.txt`

#### Notes on Dependency Choices

This project requires Python ≥ 3.9

Core dependencies include:

numpy, pandas — data handling

scikit-learn — models, PCA, PLS, cross-validation

matplotlib — plots (ROC, confusion matrices, scree plots)

joblib — model serialization

geopandas — spatial prediction maps

Some packages listed in requirements.txt (e.g. argparse, pathlib, json, typing, sys, dataclasses, re, hashlib) are part of the Python standard library and will already be available. They are included for clarity and reproducibility.

#### GeoPandas Installation (Important)

On some systems, installing geopandas via pip may fail due to compiled dependencies.

If you encounter errors, install it using conda instead:

`conda install -c conda-forge geopandas`


You can still install all other dependencies via pip.

### Step 3: Process the data 

This removes missing data and creates data structures that are easy to use. 

`python driver_scripts/process.py`

### Step 4: Run Feature Decompositions (PCA + PLS)
#### PCA
```
python driver_scripts/pca_decomp.py \
  --features-csv data/features.csv \
  --target-csv data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --variance-threshold 0.90 \
  --out-dir analysis/pca
```

Outputs:

`analysis/pca/pca_space.csv`

`analysis/pca/scree_plot.png`

`analysis/pca/pairplots.png` (unless disabled)

#### PLS
```
python driver_scripts/pls_decomp.py \
  --features-csv data/features.csv \
  --target-csv data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --variance-threshold 0.90 \
  --out-dir analysis/pls
```

Outputs:

`analysis/pls/pls_space.csv`

`analysis/pls/scree_plot.png`

`analysis/pls/pairplots.png`

### Step 5: Run Models (Leave-One-Out Cross Validation) on Raw Features vs PCA vs PLS

All LOOCV results write to:
`analysis/<space_name>/<model>/loocv/`

#### Logistic Regression (LOOCV)

##### Raw
```
python driver_scripts/logistic_regression.py \
  --mode space \
  --eval loocv \
  --x data/features.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name raw \
  --out-root analysis
```

##### PCA
```
python driver_scripts/logistic_regression.py \
  --mode space \
  --eval loocv \
  --x analysis/pca/pca_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pca \
  --out-root analysis
```

##### PLS
```
python driver_scripts/logistic_regression.py \
  --mode space \
  --eval loocv \
  --x analysis/pls/pls_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pls \
  --out-root analysis
```

#### Decision Tree (LOOCV)

##### Raw
```
python driver_scripts/decision_tree.py \
  --mode space \
  --eval loocv \
  --x data/features.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name raw \
  --out-root analysis
```

##### PCA
```
python driver_scripts/decision_tree.py \
  --mode space \
  --eval loocv \
  --x analysis/pca/pca_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pca \
  --out-root analysis
```

##### PLS
```
python driver_scripts/decision_tree.py \
  --mode space \
  --eval loocv \
  --x analysis/pls/pls_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pls \
  --out-root analysis
```

#### SVM (LOOCV)

##### Raw
```
python driver_scripts/svm_classifier.py \
  --mode space \
  --eval loocv \
  --x data/features.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name raw \
  --out-root analysis \
  --svm-type svc \
  --kernel linear \
  --C 1.0 \
  --threshold 0.0 \
  --class-weight balanced
```

#### PCA
```
python driver_scripts/svm_classifier.py \
  --mode space \
  --eval loocv \
  --x analysis/pca/pca_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pca \
  --out-root analysis \
  --svm-type svc \
  --kernel linear \
  --C 1.0 \
  --threshold 0.0 \
  --class-weight balanced
```

#### PLS
```
python driver_scripts/svm_classifier.py \
  --mode space \
  --eval loocv \
  --x analysis/pls/pls_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pls \
  --out-root analysis \
  --svm-type svc \
  --kernel linear \
  --C 1.0 \
  --threshold 0.0 \
  --class-weight balanced
```

### Step 6: Train the Final Model (fit on ALL rows)

Once you pick the best model/space from LOOCV, run the final fit so you have a production model trained on all data.

Example: SVM + PLS

```
python driver_scripts/svm_classifier.py \
  --mode space \
  --eval final \
  --x analysis/pls/pls_space.csv \
  --y data/qualification_target.csv \
  --target-col "Qualified Municipality" \
  --space-name pls \
  --out-root analysis \
  --svm-type svc \
  --kernel linear \
  --C 1.0 \
  --threshold 0.0 \
  --class-weight balanced
```

This should create:

`analysis/pls/svm/loocv/final_model.joblib`

`analysis/pls/svm/final_predictions.csv` (or similarly named output)

### Step 7: Generate a Prediction CSV for Mapping

Your map script expects a CSV that contains at least:

y_true

y_pred

plus either prob or score

Example: produce predictions from a fitted SVM model:

`python best_performing_model.py`

### Step 8: Plot the Prediction Map

Example:
```
python driver_scripts/plot_prediction_map.py \
  --raw data/TSM.xlsx \
  --excel-header 2 \
  --pred-csv analysis/pls/svm/final_predictions.csv \
  --muni-col "Municipality" \
  --target-col "Qualified Municipality" \
  --mode correctness \
  --polygon \
  --model-label "SVM + PLS (final)" \
  --out analysis/pls/svm/prediction_map.png
```