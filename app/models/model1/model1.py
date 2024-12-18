import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Создание модели CNN
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Параметры
input_shape = (64, 64, 3)  # Пример входного изображения
num_classes = 6  # Количество классов
batch_size = 32
epochs = 10

# Создание модели
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Вывод архитектуры модели
cnn_model.summary()

# Подготовка данных
data_dir = r'C:\Users\user\PycharmProjects\pythonProject\app\models\model1\dataset'

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.3  # 30% для валидации
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# Обучение модели
cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Получение предсказаний на валидационном наборе
validation_generator.reset()  # Сброс генератора
predictions = cnn_model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)  # Добавлено +1
predicted_classes = np.argmax(predictions, axis=1)

# Получаем истинные классы
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Убедимся, что длины массивов совпадают
min_length = min(len(true_classes), len(predicted_classes))
true_classes = true_classes[:min_length]
predicted_classes = predicted_classes[:min_length]

# Создание матрицы путаницы
cm = confusion_matrix(true_classes, predicted_classes)

# Визуализация матрицы путаницы
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.show()

# Сохранение модели
cnn_model.save('cnn_model.h5')
