import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Загрузка модели
model = tf.keras.models.load_model('cnn_model.h5')

# Параметры
input_shape = (64, 64, 3)  # Размер входного изображения
batch_size = 32

# Подготовка тестовых данных
test_data_dir = r'C:\Users\user\PycharmProjects\pythonProject\app\models\model1\test_dataset'

# Используем ImageDataGenerator для тестовых данных
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Прогнозирование на тестовых данных
predictions = model.predict(test_generator)

# Получение классов с максимальными вероятностями
predicted_classes = np.argmax(predictions, axis=1)

# Получение истинных классов
true_classes = test_generator.classes

# Получение названий классов
class_labels = list(test_generator.class_indices.keys())

# Вывод результатов для всех предсказаний
for i in range(len(predicted_classes)):  # Проходим по всем предсказаниям
    # Получаем индексы трех классов с наивысшими вероятностями
    top_3_indices = np.argsort(predictions[i])[-3:][::-1]
    top_3_classes = [class_labels[idx] for idx in top_3_indices]
    top_3_probs = predictions[i][top_3_indices]

    print(f"Изображение {i + 1}:")
    print(f"Истинный класс: {class_labels[true_classes[i]]}, Предсказанный класс: {class_labels[predicted_classes[i]]}")
    print(f"Топ 3 класса: {top_3_classes}, вероятности: {top_3_probs}")
    print()



