from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from visualize_neural_network.VisualizeNN import *
import tensorflow as tf
import pandas as pd
import numpy as np

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["classe"])

# "for small datasets lbdgs can converge faster and perform better" - sklearn documentation
clf = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(8, 16, 8),
                     activation="relu", verbose=False, max_iter=200)

y_train = tf.keras.utils.to_categorical(y_train -1, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test -1, num_classes=4)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8),  # Number of neurons in the first hidden layer
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(16),  # Number of neurons in the second hidden layer
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(16),  # Number of neurons in the third hidden layer
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(4),  # Output layer with 4 neurons
    tf.keras.layers.Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10000, verbose=0)

# Evaluate the model on the test set
y_pred_one_hot = model.predict(x_test)
y_pred_tf = np.argmax(y_pred_one_hot, axis=1) + 1 

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

model.summary()

# Mean squared error (MSE) and coefficient of determination (R^2) for each model
mse_1 = mean_squared_error(y_test, y_pred)
mses_w = mean_squared_error(y_train, y_pred)
mse_2 = mean_squared_error(y_test, y_pred_tf)

r2_1 = r2_score(y_test, y_pred)
r2_2 = r2_score(y_test, y_pred_tf)

print(f"Mean squared error (MSE): {mse_1:.2f}")
print(f"Mean squared error (MSE) TF: {mse_2:.2f}")
print(f"Mean squared error (MSE) ERRADO: {mses_w:.2f}")
print(f"Coefficient of determination (R^2): {r2_1:.2f}")
print(f"Mean squared error (MSE) TF: {r2_2:.2f}")
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Visualize the neural network architecture
network_structure = np.hstack((3, np.asarray(clf.hidden_layer_sizes), 4))

# Draw the Neural Network with weights
network=DrawNN(network_structure, clf.coefs_)
#network.draw()
