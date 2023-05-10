from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix


# dataset | f(aPA, pulso, espiração) = gravidade
data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["classe"], test_size=0.2)

# Modelo de decisão
regr_1 = tree.DecisionTreeClassifier(max_depth=8)
regr_1.fit(x_train, y_train)

# Predição
y_1 = regr_1.predict(x_test)

tree.export_graphviz(regr_1, out_file='decision_tree.dot', filled=True, 
                     rounded=True, feature_names=["qPA", "pulso", "respiração"], special_characters=True,
                     class_names=['estável', 'potencialmente estável', 'instável', 'crítico'])

# Mean squared error (MSE) and coefficient of determination (R^2) for each model
mse_1 = mean_squared_error(y_test, y_1)

r2_1 = r2_score(y_test, y_1)

print(f"Mean squared error (MSE): {mse_1:.2f}")
print(f"Coefficient of determination (R^2): {r2_1:.2f}")

cm = confusion_matrix(y_test, y_1)
print(f"Confusion Matrix:\n{cm}")


