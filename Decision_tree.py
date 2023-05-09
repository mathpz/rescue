from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# dataset | f(aPA, pulso, espiração) = gravidade
data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["classe"], test_size=0.05)

# Modelo de decisão
regr_1 = tree.DecisionTreeClassifier(max_depth=9)
regr_1.fit(x_train, y_train)

# Predição
y_1 = regr_1.predict(x_test)

tree.export_graphviz(regr_1, out_file='decision_tree.dot', filled=True, 
                     rounded=True, feature_names=["qPA", "pulso", "respiração"], pecial_characters=True,
                     class_names=['estável', 'potencialmente estável', 'instável', 'crítico'])

# Mean squared error (MSE) and coefficient of determination (R^2) for each model
mse_1 = mean_squared_error(y_test, y_1)

r2_1 = r2_score(y_test, y_1)

print(f"Mean squared error (MSE) - max_depth=None: {mse_1:.2f}")

print(f"Coefficient of determination (R^2) - max_depth=None: {r2_1:.2f}")

# Define a colormap
cmap = plt.get_cmap('viridis')

# Map the values of `classe` to a range of colors
colors = cmap(y_test / max(y_test))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data["qPA"], data["pulso"], data["respiração"], c=data['classe'], cmap='viridis')
#plt.scatter(y_test ,y_1, s=20, c=colors)
#ax.set_xlabel('qPA')
#ax.set_ylabel('pulso')
#ax.set_zlabel('respiração')

#plt.show()

