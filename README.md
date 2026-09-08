# decision-tree
# 🌱 Renewable Energy Adoption Prediction Using Decision Tree

## 📌 Project Overview

This project uses **Machine Learning to predict renewable energy adoption** based on environmental, energy, renewable, and economic factors.

A **Decision Tree Classifier** is used to classify whether a renewable energy technology or system will be **adopted** or **not adopted**.

The model uses four input features:

* 🌍 Carbon Emissions
* ⚡ Energy Output
* ♻️ Renewability Index
* 💰 Cost Efficiency

The model is evaluated using **Accuracy, Confusion Matrix, and Classification Report**. The trained Decision Tree is also visualized to understand how the model makes classification decisions.

---

## 🎯 Objectives

* Predict renewable energy adoption using machine learning.
* Implement a **Decision Tree Classification** algorithm.
* Use environmental, energy, renewable, and economic features for prediction.
* Evaluate the model using classification metrics.
* Visualize the confusion matrix.
* Generate and visualize the Decision Tree structure.
* Save the trained model for future predictions.

---

## 📊 Dataset

The project uses the following dataset:

```text
Renewable_Energy_Adoption.csv
```

### Input Features

| Feature                 | Description                                        |
| ----------------------- | -------------------------------------------------- |
| 🌍 `carbon_emissions`   | Carbon emissions associated with the energy system |
| ⚡ `energy_output`       | Energy output produced                             |
| ♻️ `renewability_index` | Measure of renewable energy characteristics        |
| 💰 `cost_efficiency`    | Cost-effectiveness of the energy system            |

### Target Variable

```text
adoption
```

The target represents two possible classes:

```text
0 → Non-Adoption
1 → Adoption
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and handling
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and evaluation
* **Matplotlib** – Visualization
* **Seaborn** – Confusion matrix visualization
* **Joblib** – Model saving
* **Google Colab / Jupyter Notebook**

---

## 📁 Project Structure

```text
Renewable-Energy-Adoption/
│
├── Renewable_Energy_Adoption.csv
├── renewable_energy_adoption.ipynb
├── Renewable_Energy_Adoption_model.pkl
├── dt1.png
├── README.md
└── requirements.txt
```

---

## ⚙️ Project Workflow

```text
Dataset
   ↓
Load Dataset
   ↓
Feature Selection
   ↓
Target Selection
   ↓
Train-Test Split
   ↓
Decision Tree Classifier
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Classification Report
   ↓
Decision Tree Visualization
   ↓
Save Model
```

---

# 🔬 Implementation

## 1. Import Required Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree
import joblib
```

---

## 2. Load the Dataset

```python
data = pd.read_csv(
    '/content/Renewable_Energy_Adoption.csv'
)

data.head()
```

The dataset contains the input features and the target variable used for renewable energy adoption prediction.

---

## 3. Feature and Target Selection

The following four features are selected:

```python
X = data[
    [
        'carbon_emissions',
        'energy_output',
        'renewability_index',
        'cost_efficiency'
    ]
]

y = data['adoption']
```

Where:

* **X** → Input/independent variables
* **y** → Target/dependent variable

---

## 4. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training dataset is used to train the Decision Tree, while the testing dataset is used to evaluate its performance.

---

# 🌳 Decision Tree Classifier

## 5. Create and Train the Model

A Decision Tree Classifier with a maximum depth of **3** is used:

```python
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(
    X_train,
    y_train
)
```

### Why `max_depth=3`?

The maximum depth limits the size of the tree. This helps keep the model simple and can reduce the risk of overfitting.

---

## 6. Make Predictions

The trained model predicts renewable energy adoption for the test dataset:

```python
y_pred = model.predict(
    X_test
)

print(y_pred)
```

The prediction produces the two classes:

```text
Non-Adoption
Adoption
```

---

# 📈 Model Evaluation

## 7. Accuracy Score

Accuracy measures the percentage of test samples that were classified correctly.

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

A higher accuracy generally indicates better classification performance on the test data.

---

# 📊 Confusion Matrix

A confusion matrix is used to visualize correct and incorrect predictions.

```python
conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[
        'Non-Adoption',
        'Adoption'
    ],
    yticklabels=[
        'Non-Adoption',
        'Adoption'
    ]
)

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.title(
    'Confusion Matrix'
)

plt.show()
```

### Confusion Matrix Components

| Term               | Meaning                                        |
| ------------------ | ---------------------------------------------- |
| **True Negative**  | Non-Adoption correctly predicted               |
| **True Positive**  | Adoption correctly predicted                   |
| **False Positive** | Non-Adoption incorrectly predicted as Adoption |
| **False Negative** | Adoption incorrectly predicted as Non-Adoption |

---

# 📋 Classification Report

The classification report provides detailed performance metrics:

* Precision
* Recall
* F1-score
* Support

```python
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            'Non-Adoption',
            'Adoption'
        ]
    )
)
```

### Precision

Shows how many samples predicted as a particular class were actually members of that class.

### Recall

Shows how many actual samples of a particular class were correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

---

# 🌳 Decision Tree Visualization

The trained Decision Tree can be visualized using `plot_tree()`.

```python
plt.figure(
    figsize=(12, 8)
)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[
        'Non-Adoption',
        'Adoption'
    ],
    filled=True,
    rounded=True
)

plt.savefig(
    'dt1.png'
)

plt.show()
```

The generated image:

```text
dt1.png
```

shows the decision-making structure of the trained model.

It helps understand:

* Which feature is used for each split.
* The decision conditions.
* The predicted class at each node.
* How the model reaches the final classification.

---

# 💾 Save the Trained Model

The trained Decision Tree model is saved using Joblib:

```python
joblib.dump(
    model,
    'Renewable_Energy_Adoption_model.pkl'
)
```

The saved model file is:

```text
Renewable_Energy_Adoption_model.pkl
```

It can later be loaded without retraining:

```python
model = joblib.load(
    'Renewable_Energy_Adoption_model.pkl'
)
```

---

# 🚀 Applications

This type of renewable energy adoption prediction system can be useful for:

* 🌱 Renewable energy planning
* ⚡ Energy policy analysis
* 🏙️ Smart city planning
* 🌍 Environmental decision-making
* ♻️ Sustainable energy development
* 🏭 Industrial energy planning
* 📊 Renewable technology assessment
* 💡 Energy investment decision support

---

# ✅ Advantages

* Easy to understand and interpret.
* Can model nonlinear relationships.
* Does not require feature scaling.
* Decision-making process can be visualized.
* Works with multiple input features.
* Fast prediction after training.
* Suitable for classification problems.

---

# ⚠️ Limitations

* Model performance depends on the quality of the dataset.
* A very deep tree can overfit the training data.
* The current model uses only four input features.
* Real-world renewable energy adoption depends on many additional factors.
* Predictions may not generalize well if the dataset is small or unrepresentative.

---

# 🔮 Future Improvements

The project can be improved by:

1. Increasing the size and diversity of the dataset.
2. Adding more factors such as government policies, installation cost, incentives, location, and energy demand.
3. Optimizing the Decision Tree hyperparameters.
4. Comparing Decision Tree with Random Forest, SVM, Logistic Regression, and XGBoost.
5. Using cross-validation for better model evaluation.
6. Creating a real-time renewable energy adoption prediction system.
7. Developing a web dashboard for visualization.
8. Integrating the model with an IoT-based energy monitoring system.
9. Deploying the model as a web application or API.

---

# 📌 Conclusion

This project demonstrates how a **Decision Tree Classifier can be used to predict renewable energy adoption** based on carbon emissions, energy output, renewability index, and cost efficiency.

The project covers the complete machine learning workflow, including **data loading, feature selection, train-test splitting, model training, prediction, evaluation, confusion matrix visualization, classification reporting, Decision Tree visualization, and model saving**.

The system provides a foundation for developing more advanced **AI-based renewable energy planning and sustainability decision-support systems**.

---

## 👨‍💻 Author

**Vinoth Kumar**

⭐ If you find this project useful, consider giving the repository a star!
