from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import math

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["classe"])

# Modelo de regressão
regr = RandomForestRegressor(n_estimators=int(math.sqrt(len(x_train))), max_depth=6)

regr.fit(x_train, y_train)

# Predição
y_pred = regr.predict(x_test)

# This may not the best way to view each estimator as it is small
fn=["qPA", "pulso", "respiração"]
cn=data["classe"]

for index in range(0, 5):
    tree.export_graphviz(regr.estimators_[index], out_file='decision_forest.dot',
                                    feature_names = fn, 
                                    class_names=cn,
                                    filled = True)


# Mean squared error (MSE) and coefficient of determination (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error (MSE): {mse:.2f}")
print(f"Coefficient of determination (R^2): {r2:.2f}")
