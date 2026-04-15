# Predicting Lifestyle-Related Health Outcomes Using Machine Learning

A machine learning project predicting diabetes, cardiovascular disease, and stroke risk from lifestyle and demographic factors, comparing model performance across individual and combined datasets.

Built as the final project for **DATA 5100: Foundations of Data Science** at Seattle University (Fall 2025).

---

## What It Does

Can lifestyle choices predict serious health outcomes? This project builds and evaluates machine learning models to predict three conditions: diabetes, cardiovascular disease, and stroke. We trained models on individual disease datasets and then tested whether combining them into a unified dataset improves prediction accuracy.

**Core analysis:**
- Cleaned and preprocessed three separate health outcome datasets
- Built classification models for each condition individually
- Merged datasets and evaluated whether a combined model generalizes better
- Compared model performance using accuracy, precision, recall, and F1 scores
- Generated visualizations including correlation heatmaps and feature importance rankings

---

## Project Structure

```
data5100_project/
├── Notebooks/
│   └── ...                        # Jupyter notebooks for analysis
├── Proposal-Reports/
│   └── ...                        # Project proposal and final report
├── data/
│   └── ...                        # Raw and processed health datasets
├── outputs/
│   └── ...                        # Figures, heatmaps, model results
├── src/
│   └── ...                        # Source code and utility scripts
├── LICENSE
├── README.md
└── Requirements.txt
```

---

## Tech Stack

- **Python** (pandas, NumPy, scikit-learn)
- **matplotlib** and **seaborn** (visualization)
- **Jupyter Notebook**

---

## Authors

**Ruman Sidhu** · **Devlin Hoang**
MS in Data Science, Seattle University
[GitHub](https://github.com/simplyyweirdd3) · [Email](mailto:rsidhu2@seattleu.edu)
