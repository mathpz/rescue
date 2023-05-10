from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração", "gravidade"]], data["classe"], test_size=0.2)

y_train = tf.keras.utils.to_categorical(y_train -1, num_classes=4)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(8),  
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(4),  
    tf.keras.layers.Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# Evaluate the model on the test set
y_pred_one_hot = model.predict(x_test)
y_pred_tf = np.argmax(y_pred_one_hot, axis=1) + 1 



# Mean squared error (MSE) and coefficient of determination (R^2) for each model
mse = mean_squared_error(y_test, y_pred_tf)
rmse = np.sqrt(mse)
r2_2 = r2_score(y_test, y_pred_tf)

print(f"Mean squared error (MSE): {mse:.2f} \nRoot mean squared error (RMSE): {rmse:.2f}")
print(f"R^2 (MSE): {r2_2:.2f}")
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_tf)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Crítico', 'Instável', 'p. Estável', 'Estável'])
cmd.plot()
plt.show()

