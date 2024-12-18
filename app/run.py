from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torch

app = Flask(__name__)

# Укажите директорию для сохранения загруженных изображений
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создание папки для загрузок, если она не существует
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Проверка наличия других моделей
model1_path = 'models/model1/cnn_model.h5'
model2_path = 'models/model2/dense_model.h5'

# Загрузка моделей
model1 = tf.keras.models.load_model(model1_path)
model2 = tf.keras.models.load_model(model2_path)

# Загрузка модели YOLOv5
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolov5s_results7/weights/best.pt')

# Список классов (замените на свои классы)
class_names = ['Одежда и Обувь', 'Косметика и Здоровье', 'Электроника', 'Дом и сад', 'Спорт и отдых', 'Игрушки и книги']


# Функция для предсказания класса изображения
def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_names[predicted_class_index]


@app.route('/')
def index():
    return render_template('index.html', title="Главная страница")


@app.route('/about')
def about():
    return render_template('about.html', title="О нас", class_names=class_names)


@app.route('/cnn_model', methods=['GET', 'POST'])
def cnn_model():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            predicted_class = classify_image(file_path, model1)  # Используем модель 1
        except Exception as e:
            return render_template('cnn.html', title="Модель CNN", result="Ошибка классификации: " + str(e))

        return render_template('cnn.html', title="Модель CNN", result=predicted_class)

    return render_template('cnn.html', title="Модель CNN")


@app.route('/dense_model', methods=['GET', 'POST'])
def dense_model():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            predicted_class = classify_image(file_path, model2)  # Используем модель 2
        except Exception as e:
            return render_template('dense.html', title="Модель Dense", result="Ошибка классификации: " + str(e))

        return render_template('dense.html', title="Модель Dense", result=predicted_class)

    return render_template('dense.html', title="Модель Dense")


@app.route('/yolov5_model', methods=['GET', 'POST'])
def yolov5_model():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Удаляем папку runs/detect/exp, если она существует
        exp_folder = 'runs/detect/exp'
        if os.path.exists(exp_folder):
            shutil.rmtree(exp_folder)

        # Обработка изображения с помощью YOLOv5
        yolo_model(file_path).render()
        yolo_model(file_path).save()

        result_image_name = 'png.jpg'
        result_image_path = os.path.join(exp_folder, result_image_name)

        # Move the result to the static folder
        shutil.copy(result_image_path, os.path.join(app.static_folder, 'images', result_image_name))

        # Prepare URL for the template
        result_image_url = 'images/' + result_image_name  # Adjusted to serve from static files
        return render_template('yolov5.html', title="YOLOv5 Результаты", result=result_image_url)

    return render_template('yolov5.html', title="YOLOv5")

# Добавленные API-роуты


@app.route('/api/models', methods=['GET'])
def get_models():
    models = {
        "models": [
            {"id": 1, "name": "cnn"},
            {"id": 2, "name": "dense"},
            {"id": 3, "name": "yolov5"}
        ]
    }
    return jsonify(models)


@app.route('/api/classify/cnn', methods=['POST'])
def classify_cnn():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        predicted_class = classify_image(file_path, model1)  # Используем модель 1
        return jsonify({"predicted_class": predicted_class}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/classify/dense', methods=['POST'])
def classify_dense():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        predicted_class = classify_image(file_path, model2)  # Используем модель 2
        return jsonify({"predicted_class": predicted_class}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect/yolov5', methods=['POST'])
def detect_yolov5():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Удаляем папку runs/detect/exp, если она существует
    exp_folder = 'runs/detect/exp'
    if os.path.exists(exp_folder):
        shutil.rmtree(exp_folder)

    # Обработка изображения с помощью YOLOv5
    yolo_model(file_path).render()
    yolo_model(file_path).save()

    result_image_name = 'png.jpg'
    result_image_path = os.path.join(exp_folder, result_image_name)

    shutil.copy(result_image_path, os.path.join(app.static_folder, 'images', result_image_name))

    result_image_url = 'images/' + result_image_name
    return jsonify({"result_image_url": result_image_url}), 200

if __name__ == '__main__':
    app.run(debug=True)
