from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import VisualizeNN as VisNN
import pandas as pd

data_header = ["id", "pSist", "pDiast", "qPA", "pulso", "respiração", "gravidade", "classe"]
data = pd.read_csv("treino_sinais_vitais_com_label.txt", sep=',', names=data_header, index_col=0)

x_train, x_test, y_train, y_test = train_test_split(data[["qPA", "pulso", "respiração"]], data["classe"])

# "for small datasets lbdgs can converge faster and perform better" - sklearn documentation
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 1),
                     activation="relu", verbose=True)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# Visualize the neural network architecture
