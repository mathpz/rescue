from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração", "gravidade"]], data["classe"], test_size=0.2)

# Modelo de regressão
print("Estimadores:", int(math.sqrt(len(x_train))))
regr = RandomForestClassifier(n_estimators=int(math.sqrt(len(x_train))), max_depth=4)

regr.fit(x_train, y_train)

# Predição
y_pred = regr.predict(x_test)

# Mean squared error (MSE) and coefficient of determination (R^2)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error (MSE): {mse:.2f} \n Root mean squared error (RMSE): {rmse:.2f}")
print(f"Coefficient of determination (R^2): {r2:.2f}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
cmd = ConfusionMatrixDisplay(cm, display_labels=['Crítico', 'Instável', 'Pot. Estável', 'Estável'])
cmd.plot()
plt.show()