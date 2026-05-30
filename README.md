# DecodeLabs — Project 2: Data Classification Using AI

Supervised-learning classifier built for the DecodeLabs AI Internship (Batch 2026).
The project trains a **K-Nearest Neighbors** model on the classic **Iris** dataset and
validates it with a confusion matrix and F1 score, following the deck's
**Input → Process → Output (IPO)** blueprint.

## Goal
Build a basic classification model on a small dataset: load and understand the data,
split it into training and testing sets, apply a simple classification algorithm, and
evaluate the result.

## Pipeline
| Stage | What happens |
|-------|--------------|
| **Input** | Load Iris (150 samples · 3 classes · 4 features); scale features with `StandardScaler` (mean 0, variance 1) |
| **Process** | 80/20 stratified shuffled train-test split; train `KNeighborsClassifier(n_neighbors=5)` |
| **Output** | Confusion matrix, per-class precision/recall, macro F1; bonus 5-fold cross-validated elbow plot for choosing K |

## Results
- **Accuracy:** 93.3%
- **Macro F1:** 0.933
- Setosa and versicolor classified perfectly; a couple of virginica samples overlap with versicolor (the well-known boundary between those two species).
- Cross-validated elbow confirms **K ≈ 5–6** as the optimal neighborhood size.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook DecodeLabs_Project2_Iris_KNN.ipynb
```
Then run the cells top to bottom (Shift+Enter, or Cell → Run All).

## Tech stack
Python · scikit-learn · pandas · NumPy · Matplotlib · seaborn

## Key skills demonstrated
Data handling · feature scaling · supervised-learning basics · model training,
evaluation, and hyperparameter tuning.

---
*Part of the DecodeLabs Industrial Training Kit, Batch 2026.*
