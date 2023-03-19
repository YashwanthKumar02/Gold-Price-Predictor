import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

# convert CSV file to Parquet file
gold_data = pd.read_csv('gld_price_data.csv')
gold_data.to_parquet('gld_price_data.parquet')

# load Parquet file to a Pandas DataFrame
gold_data = pd.read_parquet('gld_price_data.parquet')

X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

# Convert the Pandas DataFrame to a TensorFlow dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))

# Build the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(train_ds.batch(32), epochs=10)

# Evaluate the model
test_loss = model.evaluate(test_ds.batch(32))
print('Test Loss:', test_loss)

# Predict on Test Data
test_data_prediction = model.predict(X_test)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
