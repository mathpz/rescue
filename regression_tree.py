from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# dataset | f(aPA, pulso, espiração) = gravidade
data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade"]
data = pd.read_csv("treino_sinais_vitais_sem_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["gravidade"], test_size=0.2)

# Modelo de regressão
regr_1 = tree.DecisionTreeRegressor(max_depth=6)

regr_1.fit(x_train, y_train)

# Predição
y_1 = regr_1.predict(x_test)

tree.export_graphviz(regr_1, out_file='regression_tree.dot', 
                     feature_names=["qPA", "pulso", "respiração"],
                     class_names=True, label='all',
                     precision=2,special_characters=True,
                     filled=True, rounded=True)

# Mean squared error (MSE) and coefficient of determination (R^2) for each model
mse_1 = mean_squared_error(y_test, y_1)

r2_1 = r2_score(y_test, y_1)

print(f"Mean squared error (MSE) - max_depth=None: {mse_1:.2f}")

print(f"Coefficient of determination (R^2) - max_depth=None: {r2_1:.2f}")



