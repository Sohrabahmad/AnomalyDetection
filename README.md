# Anomaly Detection Using Machine Learning

## 📌 Project Overview

This project explores **unsupervised anomaly detection techniques** to identify unusual data points in datasets. It implements three approaches:

- **DBSCAN** — Density-based clustering that identifies clusters and noise points.
- **Isolation Forest** — An ensemble-based anomaly detection algorithm that isolates unusual observations.
- **Local Outlier Factor (LOF)** — A density-based method that detects observations with unusually low local density compared with their neighbors.

The project uses a synthetic two-moons dataset for DBSCAN and a thyroid-related dataset for Isolation Forest and LOF. It also uses Principal Component Analysis (PCA) to visualize high-dimensional data in two dimensions.

## 🎯 Objectives

- Generate and standardize synthetic non-linear data.
- Identify clusters and noise points using DBSCAN.
- Detect anomalies in the thyroid dataset using Isolation Forest.
- Compare anomaly detection results using LOF.
- Reduce high-dimensional data to two components using PCA.
- Visualize normal observations and detected outliers.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📂 Project Structure

```text
Anomaly-Detection/
│
├── Anomalis_DBSCAN.ipynb
├── Anomaly_Isolaton_Forest.ipynb
├── thyroid_dataset.csv
└── README.md
```

## 📊 Datasets

### 1. Synthetic Two-Moons Dataset

The DBSCAN notebook generates a synthetic dataset using Scikit-learn's `make_moons()` function.

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=500,
    noise=0.1,
    random_state=42
)
```

- **Samples:** 500
- **Noise level:** 0.1
- **Random state:** 42

The dataset contains two non-linear, moon-shaped groups and is useful for demonstrating density-based clustering.

### 2. Thyroid Dataset

The Isolation Forest and LOF notebook loads `thyroid_dataset.csv`.

- **Rows:** 6,916
- **Columns:** 22
- **Target/reference column:** `Outlier_label`
- **Features used for detection:** The 21 columns remaining after removing `Outlier_label`

The features include demographic information, medication indicators, clinical flags, and thyroid-related measurements such as `TSH`, `T3_measured`, `TT4_measured`, `T4U_measured`, and `FTI_measured`.

The notebook separates `Outlier_label` into `y` but does not use it to fit the anomaly detection algorithms. The models operate on the scaled feature matrix.

## ⚙️ Project Workflow

### 1. DBSCAN Anomaly Detection

**Notebook:** `Anomalis_DBSCAN.ipynb`

The notebook standardizes the generated two-moons dataset and applies DBSCAN.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=0.18,
    min_samples=5
)

labels = dbscan.fit_predict(X_scaled)
```

**Parameters:**

- `eps=0.18`: Defines the neighborhood distance.
- `min_samples=5`: Specifies the minimum neighborhood sample count used to establish a dense region.

DBSCAN assigns cluster labels to observations and uses `-1` for points identified as noise. The notebook visualizes the standardized data before and after clustering.

### 2. Data Preprocessing for Thyroid Anomaly Detection

**Notebook:** `Anomaly_Isolaton_Forest.ipynb`

The notebook loads the dataset and separates the features from the reference label.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("thyroid_dataset.csv")

X = df.drop("Outlier_label", axis=1)
y = df["Outlier_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization puts numerical features on comparable scales before applying the anomaly detection algorithms.

### 3. Isolation Forest

Isolation Forest identifies unusual observations by isolating samples through randomized feature and split selection.

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    n_estimators=200,
    contamination=0.036,
    random_state=42
)

labels = clf.fit_predict(X_scaled)
```

The model returns:

- `1` for inliers, or observations considered normal.
- `-1` for outliers, or observations flagged as anomalous.

The notebook uses PCA to project the scaled features into two dimensions and plots the resulting predictions.

### 4. Local Outlier Factor (LOF)

LOF identifies observations whose local density differs substantially from that of their neighboring points.

```python
from sklearn.neighbors import LocalOutlierFactor

neighbors = LocalOutlierFactor(
    contamination=0.036
)

labels = neighbors.fit_predict(X_scaled)
```

Like Isolation Forest, LOF returns `1` for inliers and `-1` for outliers. The notebook visualizes its predictions using the PCA projection and counts the two categories.

### 5. Dimensionality Reduction with PCA

The notebook uses PCA to visualize the high-dimensional thyroid dataset in two dimensions.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

PCA is used for visualization here; the anomaly detection models are fitted on the scaled feature matrix, not the two-dimensional PCA representation.

## 📈 Results

### Thyroid Dataset: Recorded Notebook Outputs

The notebook reports the following counts for both Isolation Forest and LOF:

| Algorithm | Normal observations (`1`) | Outliers (`-1`) | Total |
|---|---:|---:|---:|
| Isolation Forest | 6,667 | 249 | 6,916 |
| Local Outlier Factor | 6,667 | 249 | 6,916 |

The reported outlier count is approximately 3.60% of the dataset.

**Important:** These counts are the notebook's recorded outputs, not a measurement of verified clinical abnormalities. The models use a contamination setting of `0.036`, so the detected outlier proportion is influenced by that parameter. The notebook does not report a comparison of the predictions against `Outlier_label`.

For DBSCAN, the notebook produces a cluster visualization, but it does not document the final number of clusters or noise points.

## 🔍 Key Concepts

| Technique | Purpose |
|---|---|
| DBSCAN | Finds density-connected clusters and identifies noise points. |
| Isolation Forest | Detects unusual observations by isolating them. |
| Local Outlier Factor | Detects observations with unusual local density. |
| StandardScaler | Standardizes features before distance- or scale-sensitive analysis. |
| PCA | Projects high-dimensional features into fewer dimensions for visualization. |
| Contamination | Sets the expected proportion of outliers for supported anomaly detection estimators. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Anomaly-Detection
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Keep the Dataset in the Project Directory

Ensure `thyroid_dataset.csv` is in the same directory as `Anomaly_Isolaton_Forest.ipynb`, because the notebook loads it by filename.

### 4. Run the Notebooks

```bash
jupyter notebook Anomalis_DBSCAN.ipynb
```

Run the second notebook separately:

```bash
jupyter notebook Anomaly_Isolaton_Forest.ipynb
```

Execute the cells in order to reproduce the preprocessing, predictions, and visualizations.

## 🚀 Potential Applications

Anomaly detection techniques can be useful for:

- Identifying unusual records in datasets
- Detecting suspicious transactions
- Monitoring industrial sensor data
- Quality control
- Exploring medical datasets for unusual observations
- Detecting irregular patterns in operational data

The thyroid dataset example is for educational data analysis and must not be used to diagnose thyroid conditions or make clinical decisions.

## 🔮 Future Improvements

- Compare predictions with the dataset's `Outlier_label` after verifying its encoding and meaning.
- Evaluate precision, recall, and F1-score if reliable reference labels are available.
- Experiment with DBSCAN's `eps` and `min_samples` parameters.
- Compare Isolation Forest and LOF using appropriate evaluation metrics.
- Add cluster and noise counts for DBSCAN.
- Examine missing values and data types before scaling.
- Add clear legends and consistent visualization labels.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute it under a specific open-source license.
