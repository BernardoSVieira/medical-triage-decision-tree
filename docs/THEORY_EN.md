# Project Theory and Implementation

## Objective

The objective of this project is to simulate a simple medical triage system using a Decision Tree Classifier.

The system receives a set of symptoms provided by a patient and classifies the urgency level as:

- Low
- Medium
- High

This project was developed for educational purposes, aiming to apply Machine Learning concepts and understand the internal workings of Decision Trees.

---

# Dataset

The dataset used in this project was manually created and contains synthetic patient records.

Each patient is represented by six binary symptoms:

| Symptom | Description |
|----------|------------|
| Fever | The patient has a fever |
| Cough | The patient has a cough |
| Chest Pain | The patient has chest pain |
| Shortness of Breath | The patient has difficulty breathing |
| Headache | The patient has a headache |
| Nausea | The patient has nausea |

Possible values:

- 1 = Symptom present
- 0 = Symptom absent

The target variable is:

| Urgency |
|-----------|
| Low |
| Medium |
| High |

---

# Data Preprocessing

Machine Learning algorithms require numerical data to operate correctly.

Since all symptoms are already represented as binary values (0 and 1), only the urgency classification column needed to be converted.

The `LabelEncoder` class from Scikit-Learn was used to transform:

| Original Value | Encoded Value |
|---------------|--------------|
| High | 0 |
| Low | 1 |
| Medium | 2 |

> Note: The assigned numerical values may vary depending on the order in which the LabelEncoder encounters the classes.

---

# Why Use a Decision Tree?

Decision Trees are among the most intuitive Machine Learning algorithms.

Their behavior resembles a sequence of questions and answers.

Simplified example:

```text
Chest Pain?

├── Yes → High Urgency
└── No
    │
    ├── Fever?
    │   ├── Yes → Medium Urgency
    │   └── No → Low Urgency
```

The algorithm automatically learns which questions are most important for separating the data into different categories.

---

# Entropy

This project uses:

```python
criterion="entropy"
```

Entropy is a measure of disorder or uncertainty within a dataset.

### Low Entropy

When all examples belong to the same class.

Example:

```text
High
High
High
High
```

In this case, uncertainty is low.

### High Entropy

When different classes are mixed together.

Example:

```text
High
Low
Medium
High
Low
```

In this case, uncertainty is high.

---

# Information Gain

The goal of a Decision Tree is to reduce entropy as much as possible.

To achieve this, the algorithm evaluates each symptom and determines which one best separates the data.

This reduction in entropy is called **Information Gain**.

The symptom that provides the highest Information Gain is selected first in the tree.

For example:

```text
Chest Pain
```

may be considered more important than:

```text
Nausea
```

if it separates urgency levels more effectively.

---

# Train/Test Split

To verify whether the model has learned useful patterns, the dataset is divided into two parts:

- 70% for training
- 30% for testing

Using:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

## Training Data

Used to teach the model.

## Testing Data

Used to evaluate the model on examples it has never seen before.

This helps prevent the model from simply memorizing the dataset.

---

# Model Training

After preprocessing, a classifier is created:

```python
clf = DecisionTreeClassifier(
    criterion="entropy"
)
```

The model is then trained:

```python
clf.fit(X_train, y_train)
```

At this stage, the Decision Tree analyzes all training examples and automatically builds its decision structure.

---

# Model Evaluation

After training, it is necessary to evaluate whether the model is performing correctly.

Two primary metrics were used:

- Accuracy
- Confusion Matrix

---

## Accuracy

Accuracy measures the percentage of correct predictions made by the model.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

Example:

```text
23 correct predictions out of 24 cases
```

Result:

```text
Accuracy = 95.8%
```

The closer the value is to 100%, the better the model's performance.

---

# Confusion Matrix

The Confusion Matrix provides a detailed view of where the model succeeds and where it makes mistakes.

Example:

```text
[[12 0 0]
 [ 0 4 1]
 [ 0 0 7]]
```

Considering the class order:

```text
High
Low
Medium
```

The interpretation would be:

- 12 High urgency cases correctly classified
- 4 Low urgency cases correctly classified
- 1 Low urgency case incorrectly classified as Medium
- 7 Medium urgency cases correctly classified

The matrix helps identify exactly which classes are being confused.

---

# Interactive Triage System

After the model is trained, the user can perform a medical triage directly through the terminal.

Example:

```text
Does the patient have Fever? (Y/N)
Does the patient have Cough? (Y/N)
Does the patient have Chest Pain? (Y/N)
Does the patient have Shortness of Breath? (Y/N)
Does the patient have Headache? (Y/N)
Does the patient have Nausea? (Y/N)
```

The answers are converted into binary values:

```text
Y → 1
N → 0
```

A new record is then created:

```python
[1, 0, 1, 1, 0, 0]
```

which is passed to the model through:

```python
clf.predict()
```

The system then returns the predicted urgency level.

---

# Decision Tree Visualization

The project also allows visualization of the generated Decision Tree.

The following function is used:

```python
plot_tree()
```

This visualization helps understand:

- Which symptoms are most important;
- How decisions are made;
- How patients are classified into urgency levels.

---

# Project Limitations

This project is intended solely for educational purposes.

The dataset is synthetic and was manually created for study and experimentation.

Therefore:

- It should not be used for real medical diagnoses;
- It does not replace healthcare professionals;
- It does not represent official medical protocols.

---

# Concepts Applied

The following concepts were applied during the development of this project:

- Data manipulation with Pandas
- Machine Learning with Scikit-Learn
- Decision Trees
- Entropy
- Information Gain
- Label Encoding
- Train/Test Split
- Accuracy
- Confusion Matrix
- Model visualization
- Terminal-based user interaction
- Python project organization
- Version control with Git and GitHub

---

# Learning Outcomes

During the development of this project, it was possible to understand in practice:

- How to prepare a dataset for training;
- How a Decision Tree makes decisions;
- How to evaluate the quality of a Machine Learning model;
- How to transform user input into predictions;
- How to integrate a trained model into an interactive Python application.

The primary goal of this project was not only to obtain accurate predictions, but also to understand the complete process of building, training, and evaluating a Machine Learning model.

---

# References and Credits

This project was developed as a practical study activity focused on Decision Trees and Machine Learning.

The primary reference used for the initial implementation and conceptual understanding was the material provided by **Professor Saulo Popov Zambiasi** through the ARISA Wiki:

- https://wiki.arisa.com.br/index.php?title=%C3%81rvores_de_Decis%C3%A3o_em_Python

Additional resources on Decision Trees, Entropy, Information Gain, and supervised classification were consulted to deepen theoretical understanding.

---

# Acknowledgements

Special thanks to **Professor Saulo Popov Zambiasi** for providing the educational material that served as the foundation for this project.

The project was expanded beyond the original example and includes:

- A custom medical triage dataset;
- Urgency classification (Low, Medium, and High);
- Train/Test Split implementation;
- Accuracy evaluation;
- Confusion Matrix analysis;
- Decision Tree visualization;
- Interactive terminal-based symptom input;
- GitHub-ready project organization and documentation.
