from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from visualize_neural_network.VisualizeNN import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_sem_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["gravidade"], test_size=0.2)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(32),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(16),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(8),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(1),  
    tf.keras.layers.Activation('linear')
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse', 'mae'])


# Train the model
history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean squared error (MSE): {mse:.2f} \n Root mean squared error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
# Plot the predictions against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.xlim(left=0)
plt.ylabel("Predicted")
plt.show()

