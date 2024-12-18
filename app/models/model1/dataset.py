import os


def count_images_in_directory(directory):
    # Список расширений изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # Проверяем, существует ли директория
    if not os.path.exists(directory):
        print(f"Директория '{directory}' не найдена.")
        return

    # Проходим по всем поддиректориям и файлам в указанной директории
    for root, dirs, files in os.walk(directory):
        image_count = 0  # Счетчик изображений для текущей директории
        for filename in files:
            # Получаем расширение файла
            _, ext = os.path.splitext(filename)
            # Проверяем, является ли файл изображением
            if ext.lower() in image_extensions:
                image_count += 1

        # Выводим количество изображений в текущей директории
        if image_count > 0:
            print(f"Количество изображений в директории '{root}': {image_count}")


# Укажите путь к директории, которую хотите проверить
directory_path = r'C:\Users\user\PycharmProjects\pythonProject\app\models\dataset'
count_images_in_directory(directory_path)
