#  Iris Flower Classification


##  Project Overview
This project applies machine learning techniques to classify iris flowers into three species: **Setosa, Versicolor, and Virginica**.

It demonstrates the complete machine learning workflow, including data analysis, visualization, model training, and evaluation.

---

##  Dataset Details
- Total Samples: 150
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target:
  - Species (3 classes)

---

## Exploratory Data Analysis
- Pairplot to visualize feature relationships
- Distribution plots for feature understanding
- Correlation heatmap to identify important features

---

##  Machine Learning Models
The following models were trained and compared:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

---

##  Model Evaluation
Models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## Results
- All models performed well due to clean dataset
- Random Forest & Logistic Regression gave the best performance
- KNN performance varies based on K value

---

##  Key Learnings
- Understanding classification problems
- Importance of feature relationships
- Model comparison techniques
- Practical implementation of ML algorithms

---

##  Future Improvements
- Hyperparameter tuning (GridSearchCV)
- Try advanced models (SVM, Neural Networks)
- Deploy as a web app using Streamlit

---

##  Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

--

##  Live Demo
(Add Streamlit link after deployment)


##  How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
