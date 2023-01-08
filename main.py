
# Understanding Neural Networks with Iris Dataset - Code
# Author: Lewis Watson

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

import wandb
from wandb.integration.keras import WandbMetricsLogger
wandb.init(project="Iris-Classifier")
wandb.config.dropout = 0.2
wandb.config.batch_size = 12
wandb.config.epochs = 200

# Load the data from the CSV files, each row is a 10 element list of features
# and a 1 element list of labels
df = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# extract the last column as the labels
labels = df.pop('species')

# convert the labels to one-hot encoding
labels = pd.get_dummies(labels)

# convert the dataframes to numpy arrays
X = df.values

print(X)
y = labels.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(4,)))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with batch size of 10 and 200 epochs, and early stopping
model.fit(X_train, y_train,
          batch_size=12,
          epochs=200,
          validation_data=(X_val, y_val),
          callbacks=[WandbMetricsLogger()])


# Evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)
print('Test size:', len(X_test))