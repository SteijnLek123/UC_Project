import osmnx as ox
import matplotlib as plt
import geopandas as gpd
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
from sklearn.svm import SVC
from sklearn import svm

from sklearn.metrics import classification_report

Porto_embeddings = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\PortoEmbeddings.csv')


def svm_classification(dataframe, target_column):
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a SVM Classifier with Linear Kernel
    clf = svm.SVC(kernel='linear')

    # Train the model using the training sets
    clf.fit(X_train_scaled, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report

# Replace 'your_dataframe' with the actual DataFrame variable and 'crossing_boolean' with the actual column name.
# accuracy, report = svm_classification(your_dataframe, 'crossing_boolean')
# print(f"Accuracy: {accuracy}\n")
# print(f"Classification Report:\n{report}")
print(svm_classification(Porto_embeddings,'crossing_boolean'))