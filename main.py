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

#Clustering###
group_size = 30
clf.fit_predict(Porto_data.values)
clusters = clf.labels_
Porto_data['cluster'] = clusters

X1 = Porto_data[['element_type', 'highway']].values
X2 = Porto_data[['element_type', 'n_type']].values
X3 = Porto_data[['highway', 'n_type']].values
X_combined = np.concatenate([X1, X2, X3], axis=1)

Y1 = Porto_data['n_type'].values - 1 
Y2 = Porto_data['highway'].values
Y3 = Porto_data['element_type'].values

X_train, X_test, Y_train, Y_test = train_test_split(X_combined, np.column_stack((Y1, Y2, Y3)), test_size=0.2, random_state=42)

input_layer = Input(shape=(6,))
shared_layers = Dense(64, activation='relu')(input_layer)
shared_layers = Dense(32, activation='relu')(shared_layers)



output_task1 = Dense(5, activation='softmax', name='output_task1')(shared_layers)
output_task2 = Dense(9, activation='softmax', name='output_task2')(shared_layers)
output_task3 = Dense(2, activation='softmax', name='output_task3')(shared_layers)

model = Model(inputs=input_layer, outputs=[output_task1, output_task2, output_task3])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'output_task1': 'sparse_categorical_crossentropy',
                    'output_task2': 'sparse_categorical_crossentropy',
                    'output_task3': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

model.fit(X_train, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]], epochs=10, batch_size=32, validation_split=0.2)
performance = model.evaluate(X_test, [Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]], verbose=0)


embeddings = model.get_layer(index=-3).get_weights()[0]
