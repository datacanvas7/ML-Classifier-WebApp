import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

sns.get_dataset_names()

# App Heading
st.write("""
# Explore different ML models and datasets
This WebApp is developed using Streamlit library
""")

# Page Layout
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# Load dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

# Overview dataset
X, y = get_dataset(dataset_name)
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max.depth # depth of every tree in Random Forest
        params["n_estimators"] = n_estimators # Number of trees
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                           max_depth=params["max_depth"], random_state=1234)
    return clf

# For Code Visiblity Checkbox
if st.checkbox("Show Code"):
  with st.echo():
    # CLF Function
      clf = get_classifier(classifier_name, params)
      # Splitting traing/testing dataset
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
      # Training classifier
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      # Model Accuracy
      acc = accuracy_score(y_test, y_pred)

# CLF Function
clf = get_classifier(classifier_name, params)

# Splitting traing/testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Training classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Model evaluation
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# Plot dataset
pca = PCA(2)
x_projected = pca.fit_transform(X)

# Slicing data
x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)

