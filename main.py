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
from sklearn.decomposition import PCA




# load in Data
Porto_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\crossing_porto')
SanFrancisco_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\crossing_san_francisco')
Tokyo_data = pd.read_csv(r'C:\Users\NimaMehrafar\UC\UC_Project\crossing_tokyo')


def extract_lon_lat(point_string):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", point_string)
    return float(nums[0]), float(nums[1])

#Preprocessing
def preprocess(dataframe):
      label_encoder = LabelEncoder()
      dataframe['highway_tag'] = label_encoder.fit_transform(dataframe['highway_tag'])
      dataframe['element_type'] = label_encoder.fit_transform(dataframe['element_type'])
      dataframe['geometry_float'] = dataframe['geometry'].apply(lambda x: pd.Series(extract_lon_lat(x))).prod(axis=1)
      dataframe = dataframe[['element_type','highway_tag','geometry_float']]

      return dataframe

Porto_data_p = preprocess(Porto_data)

clf_Porto = KMeansConstrained(
     n_clusters=int(round(len(Porto_data)/100)),
    size_min=90,
    size_max=100, 
    random_state=0)

#Clustering###
clf_Porto.fit_predict(Porto_data_p.values)
clusters = clf_Porto.labels_
Porto_data_p['cluster'] = clusters


def apply_pca_and_clustering(data, clf):
    data['cluster'] = clf.fit_predict(data[['element_type','highway_tag','geometry_float']].values)
    
    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data[['element_type','highway_tag','geometry_float']])
    
    # Concatenate the cluster labels with the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = data['cluster']
    
    # Plot the PCA results
    plt.figure(figsize=(10, 7))
    plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['cluster'], cmap='viridis', marker='o')
    plt.title('PCA of Porto Data by Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


def NeuralNet(len_highway, len_element, dataframe):
      dataframe = dataframe[['element_type','highway_tag','geometry_float']]
      X1 = dataframe[['element_type', 'geometry_float']].values
      X2 = dataframe[['highway_tag', 'geometry_float']].values
      X3 = dataframe[['element_type', 'highway_tag']].values
      X_combined = np.concatenate([X1, X2, X3], axis=1)

      Y1 = dataframe['highway_tag'].values
      Y2 = dataframe['element_type'].values 
      Y3 = dataframe['geometry_float'].values

      X_train, X_test, Y_train, Y_test = train_test_split(X_combined, np.column_stack((Y1, Y2, Y3)), test_size=0.2, random_state=42)

      input_layer = Input(shape=(6,))
      shared_layers = Dense(64, activation='relu')(input_layer)
      shared_layers = Dense(64, activation='relu',name='finaldense')(input_layer)

      output_task1 = Dense(len_highway, activation='softmax', name='task1')(shared_layers)
      output_task2 = Dense(len_element, activation='softmax', name='task2')(shared_layers)
      output_task3 = Dense(1, activation='linear', name='task3')(shared_layers)

      model = Model(inputs=input_layer, outputs=[output_task1, output_task2, output_task3])

      model.compile(optimizer='adam',
                  loss={
                        'task1': 'sparse_categorical_crossentropy',
                        'task2': 'sparse_categorical_crossentropy',
                        'task3': 'mean_squared_error'
                  },
                  metrics={
                        'task1': 'accuracy',
                        'task2': 'accuracy',
                        'task3': 'mean_squared_error' 
                  })
      model.fit(X_train, [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]], epochs=50, batch_size=32, validation_split=0.2)
      performance = model.evaluate(X_test, [Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]], verbose=0)
      embedding_model = Model(inputs=model.input, outputs=model.get_layer('finaldense').output)
      embeddings = embedding_model.predict(X_combined)
      return embeddings


all_embeddings = pd.DataFrame()

for cluster in Porto_data_p['cluster'].unique():
    cluster_data = Porto_data_p[Porto_data_p['cluster'] == cluster]
    embeddings = NeuralNet(cluster_data['highway_tag'].unique(), cluster_data['element_type'].unique(), cluster_data)
    embeddings_df = pd.DataFrame(embeddings, index=cluster_data.index)
    embeddings_df['cluster'] = cluster
    all_embeddings = pd.concat([all_embeddings, embeddings_df])

