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



# load in Data
Porto_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\porto_data_2')
SanFrancisco_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\san_francisco_data_2')
Tokyo_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\tokyo_data_2')

clf = KMeansConstrained(
     n_clusters=85,
    size_min=40,
    size_max=60, 
    random_state=0)

#Preprocessing
Porto_data = Porto_data[['element_type','highway','n_type']]
label_encoder = LabelEncoder()
Porto_data['highway'] = label_encoder.fit_transform(Porto_data['highway'])
Porto_data['element_type'] = label_encoder.fit_transform(Porto_data['element_type'])

#Clustering
group_size = 30
clf.fit_predict(Porto_data.values)
clusters = clf.labels_
Porto_data['cluster'] = clusters

# X and Y
X = Porto_data[['element_type','highway']]
y = Porto_data['n_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)