import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Создание полносвязной модели
def create_dense_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))  # Преобразуем 2D изображения в 1D массив
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Параметры
input_shape = (64, 64, 3)  # Размер входного изображения
num_classes = 6  # Количество классов
batch_size = 32  # Размер батча
epochs = 10  # Количество эпох

# Создание модели
dense_model = create_dense_model(input_shape, num_classes)
dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Вывод архитектуры модели
dense_model.summary()

# Подготовка данных
data_dir = r'C:\Users\user\PycharmProjects\pythonProject\app\models\model1\dataset'

# Создание генераторов данных
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.3)

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

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Обучение модели
dense_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Оценка модели на тестовых данных
test_loss, test_accuracy = dense_model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test accuracy: {test_accuracy:.4f}")

# Сохранение доли ошибки в файл
with open('error_rate_dense.txt', 'w') as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")

# Предсказание классов для тестового набора
test_generator.reset()  # Сброс генератора
predictions = dense_model.predict(test_generator, steps=test_generator.samples // batch_size + 1)  # Добавлено +1 для полного захвата
predicted_classes = np.argmax(predictions, axis=1)

# Получение истинных классов
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Проверка длины классов и подгонка
min_length = min(len(true_classes), len(predicted_classes))
true_classes = true_classes[:min_length]
predicted_classes = predicted_classes[:min_length]

# Создание матрицы путаницы
cm = confusion_matrix(true_classes, predicted_classes)

# Визуализация матрицы путаницы
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Матрица ошибок для Dense модели')
plt.show()

# Генерация отчета по классификации
if len(true_classes) == len(predicted_classes):
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)

    # Сохранение отчета в файл
    with open('classification_report_dense.txt', 'w') as f:
        f.write(report)
else:
    print("Warning: The lengths of true classes and predicted classes do not match!")

# Сохранение модели
dense_model.save('dense_model.h5')
