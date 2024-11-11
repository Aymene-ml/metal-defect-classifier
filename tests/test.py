import requests
# test the API
resp = requests.post('http://localhost:5000/api/predict',files={'file':open('scratch.jpg', 'rb')})

print(resp.text)