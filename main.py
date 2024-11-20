import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Modelo:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, header=None, names=names)
        print("Dataset carregado com sucesso.")

    def TratamentoDeDados(self):
        print("Iniciando tratamento de dados.")
        # Codificar rótulos de espécies
        label_encoder = LabelEncoder()
        self.df['Species'] = label_encoder.fit_transform(self.df['Species'])

        # Normalizar os dados
        scaler = StandardScaler()
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.df[features] = scaler.fit_transform(self.df[features])

        print("Dados tratados com sucesso.")

    def Treinamento(self):
        print("Iniciando treinamento dos modelos.")
        # Dividir dados em treino e teste
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Modelo SVM
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm_model

        # Modelo de Regressão Linear
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        self.models['LinearRegression'] = lr_model

        print("Modelos treinados com sucesso.")

    def Teste(self):
        print("Iniciando testes dos modelos.")
        results = {}
        for name, model in self.models.items():
            if name == 'LinearRegression':
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                results[name] = f"MSE: {mse:.4f}"
            else:
                y_pred = model.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                results[name] = f"Accuracy: {acc:.4f}"

        print("Resultados dos testes:")
        for model, result in results.items():
            print(f"{model}: {result}")

    def Train(self):
        self.CarregarDataset("iris.data")
        self.TratamentoDeDados()
        self.Treinamento()
        self.Teste()

# Exemplo de uso
modelo = Modelo()
# modelo.Train()  # Descomente para executar quando o arquivo "iris.data" estiver disponível no ambiente local.
