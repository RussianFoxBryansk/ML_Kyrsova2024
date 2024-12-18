import requests

# Тестирование GET-запроса
response = requests.get('http://127.0.0.1:5000/api/models')
print(response.json())

# Тестирование POST-запроса
files = {'image': open('uploads/00000042.jpg', 'rb')}
response = requests.post('http://127.0.0.1:5000/api/classify/cnn', files=files)
print(response.json())
