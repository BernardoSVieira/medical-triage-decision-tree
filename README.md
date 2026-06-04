# 🌳 Medical Triage Decision Tree

<div align="center">

🇺🇸 <a href="README.md">English</a> •
🇧🇷 <a href="README_PTBR.md">Português</a>

</div>

A Machine Learning project that simulates a medical triage system using a Decision Tree Classifier.

The model receives a set of patient symptoms and predicts the urgency level:

- 🟢 Low Urgency
- 🟡 Medium Urgency
- 🔴 High Urgency

This project was developed for educational purposes to explore Decision Trees, Information Gain (Entropy), model evaluation, and classification systems using Scikit-Learn.

---

## Features

- Synthetic dataset generation
- Decision Tree training using Information Gain (Entropy)
- Model evaluation with:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Tree visualization
- Interactive command-line triage system

---

## Symptoms Used

The model analyzes the following symptoms:

| Symptom | Type |
|----------|----------|
| Fever | Binary |
| Cough | Binary |
| Chest Pain | Binary |
| Shortness of Breath | Binary |
| Headache | Binary |
| Nausea | Binary |

---

## Urgency Levels

| Value | Classification |
|---------|---------|
| 0 | Low |
| 1 | Medium |
| 2 | High |

---

## Technologies

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
