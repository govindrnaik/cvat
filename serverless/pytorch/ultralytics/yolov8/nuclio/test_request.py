import base64
import json
import requests

with open("/home/ubuntu/cvat/original_image (17).png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

data = {"image": image_data}

response = requests.post("http://localhost:32769", json=data)

print(response.json())
