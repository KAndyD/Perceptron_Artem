import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


class ArtemPerceptron:
    def __init__(self, input_size, hidden_size=20, learning_rate=0.01, epochs=1000):
        self.name = "Артем"
        self.lr = learning_rate
        self.epochs = epochs

        # Веса и смещения для слоя 1 (input -> hidden)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # (30,20)
        self.b1 = np.zeros((1, hidden_size))  # (1,20)

        # Веса и смещения для слоя 2 (hidden -> output)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01  # (20,1)
        self.b2 = np.zeros((1, 1))  # (1,1)

        self.losses = []
        self.accuracies = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.X = X  # (N,30)
        self.z1 = np.dot(X, self.W1) + self.b1  # (N,20)
        self.a1 = self.relu(self.z1)             # (N,20)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (N,1)
        self.a2 = self.sigmoid(self.z2)          # (N,1)
        return self.a2

    def compute_loss(self, y_hat, y):
        epsilon = 1e-9
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        y = y.reshape(-1,1)  # (N,1)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true) * 100

    def train(self, X, y, target_accuracy=85.0):
        y = y.reshape(-1, 1)  # (N,1)
        print(f"Перцептрон {self.name} начинает обучение.\n")

        for epoch in range(1, self.epochs + 1):
            # Прямой проход
            y_hat = self.forward(X)  # (N,1)

            # Ошибка
            error = y_hat - y  # (N,1)

            # Градиенты для выходного слоя
            dZ2 = error  # (N,1), т.к. BCE + sigmoid -> dL/dZ = y_hat - y
            dW2 = np.dot(self.a1.T, dZ2) / len(X)  # (20,N)*(N,1) = (20,1)
            db2 = np.mean(dZ2, axis=0, keepdims=True)  # (1,1)

            # Градиенты для скрытого слоя
            dA1 = np.dot(dZ2, self.W2.T)  # (N,1)*(1,20) = (N,20)
            dZ1 = dA1 * self.relu_derivative(self.z1)  # (N,20)

            dW1 = np.dot(X.T, dZ1) / len(X)  # (30,N)*(N,20) = (30,20)
            db1 = np.mean(dZ1, axis=0, keepdims=True)  # (1,20)

            # Обновление весов
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            loss = self.compute_loss(y_hat, y)
            y_pred = (y_hat >= 0.5).astype(int).flatten()
            acc = self.accuracy(y_pred, y.flatten())

            self.losses.append(loss)
            self.accuracies.append(acc)

            if epoch % 50 == 0 or epoch == 1:
                print(f"Эпоха {epoch:4d}: Потери = {loss:.6f}, Точность = {acc:.2f}%")

            if acc >= target_accuracy:
                print(f"\nДостигнута точность {acc:.2f}% — обучение завершено на эпохе {epoch}! Артем теперь очень умный)))\n")
                break
        else:
            print(f"\nДостигнут лимит в {self.epochs} эпох — обучение остановлено. Артем все еще тупой(((\n")

    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat >= 0.5).astype(int).flatten()

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Потери', color='blue')
        plt.title(f'График потерь — {self.name}')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label='Точность', color='green')
        plt.title(f'График точности — {self.name}')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("artem_metrics.png")
        plt.show()

    def save_report_csv(self, filename="report.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Эпоха", "Потери", "Точность (%)"])
            for epoch, (loss, acc) in enumerate(zip(self.losses, self.accuracies), start=1):
                writer.writerow([epoch, loss, acc])
        print(f"Отчёт сохранён в файл {filename}")


def generate_data(samples=100000, features=30):
    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=25,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.01,
        class_sep=2.0
    )
    return X, y


if __name__ == "__main__":
    X, y = generate_data(samples=100000, features=30)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = ArtemPerceptron(input_size=30, hidden_size=20, learning_rate=0.01, epochs=1000)
    model.train(X_train, y_train, target_accuracy=85.0)

    y_pred = model.predict(X_test)
    acc = model.accuracy(y_pred, y_test)
    print(f"\nТочность на тестовой выборке: {acc:.2f}%")

    model.plot_metrics()
    model.save_report_csv()

