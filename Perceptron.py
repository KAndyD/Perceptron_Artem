import numpy as np  # Подключение библиотеки NumPy для работы с массивами и матрицами
import matplotlib  # Подключение библиотеки Matplotlib для визуализации данных
matplotlib.use('TkAgg')  # Установка backend 'TkAgg' для отображения графиков в отдельных окнах
import matplotlib.pyplot as plt  # Подключение модуля pyplot для построения графиков
import csv  # Подключение модуля csv для чтения и записи файлов в формате CSV
from sklearn.datasets import make_classification  # Подключение генератора синтетических классификационных данных
from sklearn.preprocessing import StandardScaler  # Подключение инструмента для нормализации признаков


class ArtemPerceptron:  # Определение класса перцептрона с именем "Артем"
    def __init__(self, input_size, hidden_size=20, learning_rate=0.01, epochs=1000):
        self.name = "Артем"  # Название модели

        self.lr = learning_rate  # Скорость обучения
        self.epochs = epochs  # Количество эпох обучения

        # Веса и смещения для слоя 1 (входной -> скрытый)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Инициализация весов слоя 1 маленькими случайными числами
        self.b1 = np.zeros((1, hidden_size))  # Инициализация смещений слоя 1 нулями

        # Веса и смещения для слоя 2 (скрытый -> выходной)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01  # Инициализация весов слоя 2 маленькими случайными числами
        self.b2 = np.zeros((1, 1))  # Инициализация смещения слоя 2 нулями

        self.losses = []  # Хранение значений ошибки (loss) по эпохам
        self.accuracies = []  # Хранение значений точности (accuracy) по эпохам

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Сигмоидная функция активации

    def sigmoid_derivative(self, a):
        return a * (1 - a)  # Производная сигмоиды по её выходу

    def relu(self, z):
        return np.maximum(0, z)  # ReLU-функция активации (обнуляет отрицательные значения)

    def relu_derivative(self, z):
        return (z > 0).astype(float)  # Производная ReLU (0 для z <= 0, иначе 1)

    def forward(self, X):
        self.X = X  # Входной слой (размерность N x входные признаки)
        self.z1 = np.dot(X, self.W1) + self.b1  # Линейное преобразование входа для скрытого слоя
        self.a1 = self.relu(self.z1)  # Применение ReLU к скрытому слою
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Линейное преобразование скрытого слоя для выходного слоя
        self.a2 = self.sigmoid(self.z2)  # Применение сигмоиды к выходу
        return self.a2  # Возвращение выходных вероятностей

    def compute_loss(self, y_hat, y):
        epsilon = 1e-9  # Малое значение для избежания деления на ноль и логарифма от нуля
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Ограничение y_hat в интервале (0,1)
        y = y.reshape(-1, 1)  # Преобразование y в вектор-столбец
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))  # Бинарная кросс-энтропия (функция потерь)

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true) * 100  # Вычисление точности предсказаний в процентах

    def train(self, X, y, target_accuracy=85.0):
        y = y.reshape(-1, 1)  # Преобразование целевого вектора y в столбец (N,1)
        print(f"Перцептрон {self.name} начинает обучение.\n")

        for epoch in range(1, self.epochs + 1):
            y_hat = self.forward(X)  # Прямой проход: получение выходных предсказаний (N,1)

            error = y_hat - y  # Вычисление ошибки предсказания (разность между y_hat и истинными y)

            dZ2 = error  # Градиент функции потерь по z2 (для сигмоиды и BCE: dL/dZ = y_hat - y)
            dW2 = np.dot(self.a1.T, dZ2) / len(X)  # Градиент по весам скрытого-выходного слоя
            db2 = np.mean(dZ2, axis=0, keepdims=True)  # Градиент по смещению выходного слоя

            dA1 = np.dot(dZ2, self.W2.T)  # Градиент ошибки на скрытом слое
            dZ1 = dA1 * self.relu_derivative(self.z1)  # Применение производной ReLU

            dW1 = np.dot(X.T, dZ1) / len(X)  # Градиент по весам входного-скрытого слоя
            db1 = np.mean(dZ1, axis=0, keepdims=True)  # Градиент по смещению скрытого слоя

            self.W2 -= self.lr * dW2  # Обновление весов выходного слоя
            self.b2 -= self.lr * db2  # Обновление смещений выходного слоя
            self.W1 -= self.lr * dW1  # Обновление весов скрытого слоя
            self.b1 -= self.lr * db1  # Обновление смещений скрытого слоя

            loss = self.compute_loss(y_hat, y)  # Вычисление функции потерь
            y_pred = (y_hat >= 0.5).astype(int).flatten()  # Преобразование вероятностей в метки классов
            acc = self.accuracy(y_pred, y.flatten())  # Вычисление точности

            self.losses.append(loss)  # Сохранение потерь текущей эпохи
            self.accuracies.append(acc)  # Сохранение точности текущей эпохи

            if epoch % 50 == 0 or epoch == 1:
                print(f"Эпоха {epoch:4d}: Потери = {loss:.6f}, Точность = {acc:.2f}%")  # Логирование прогресса

            if acc >= target_accuracy:
                print(
                    f"\nДостигнута точность {acc:.2f}% — обучение завершено на эпохе {epoch}! Артем теперь очень умный)))\n")
                break  # Завершение обучения при достижении целевой точности
        else:
            print(
                f"\nДостигнут лимит в {self.epochs} эпох — обучение остановлено. Артем все еще тупой(((\n")  # Сообщение при недостижении цели

    def predict(self, X):
        y_hat = self.forward(X)  # Получение вероятностей через forward
        return (y_hat >= 0.5).astype(int).flatten()  # Преобразование в метки классов

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))  # Размер графика

        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Потери', color='blue')  # График потерь
        plt.title(f'График потерь — {self.name}')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label='Точность', color='green')  # График точности
        plt.title(f'График точности — {self.name}')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()  # Автоматическая подгонка размещения
        plt.savefig("artem_metrics.png")  # Сохранение графика в файл
        plt.show()  # Отображение графика

    def save_report_csv(self, filename="report.csv"):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)  # Инициализация CSV-писателя
            writer.writerow(["Эпоха", "Потери", "Точность (%)"])  # Заголовок
            for epoch, (loss, acc) in enumerate(zip(self.losses, self.accuracies), start=1):
                writer.writerow([epoch, loss, acc])  # Запись результатов по эпохам
        print(f"Отчёт сохранён в файл {filename}")  # Подтверждение сохранения

def generate_data(samples=1000000, features=30):
    X, y = make_classification(
        n_samples=samples,  # Количество примеров
        n_features=features,  # Количество признаков
        n_informative=25,  # Количество информативных признаков
        n_redundant=5,  # Количество избыточных признаков
        n_classes=2,  # Количество классов (двоичная классификация)
        random_state=42,  # Фиксированное зерно генератора
        flip_y=0.01,  # Доля шумных меток
        class_sep=2.0  # Разделимость классов
    )
    return X, y  # Возврат признаков и меток


if __name__ == "__main__":
    X, y = generate_data(samples=1000000, features=30)  # Генерация обучающих данных

    scaler = StandardScaler()  # Инициализация стандартизации признаков
    X = scaler.fit_transform(X)  # Масштабирование данных к нулевому среднему и единичному отклонению

    split = int(0.8 * len(X))  # Определение размера обучающей выборки (80%)
    X_train, y_train = X[:split], y[:split]  # Формирование обучающей выборки
    X_test, y_test = X[split:], y[split:]  # Формирование тестовой выборки

    model = ArtemPerceptron(input_size=30, hidden_size=20, learning_rate=0.01, epochs=1000)  # Создание экземпляра перцептрона
    model.train(X_train, y_train, target_accuracy=85.0)  # Обучение модели на обучающей выборке

    y_pred = model.predict(X_test)  # Получение предсказаний на тестовой выборке
    acc = model.accuracy(y_pred, y_test)  # Вычисление точности модели на тестовой выборке
    print(f"\nТочность на тестовой выборке: {acc:.2f}%")  # Вывод точности в консоль

    model.plot_metrics()  # Построение и отображение графиков потерь и точности
    model.save_report_csv()  # Сохранение отчёта об обучении в CSV-файл

