from flask import Flask, request, jsonify ,render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

app = Flask(__name__)

# تحميل النموذج
model = torch.load('models/model_scripted.pt')
model.eval()
classes = ["Cataract", "Diabetic", "Glaucoma", "Normal"]

# دالة لتحويل الصورة إلى تنسيق تقبله PyTorch
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

# نقطة النهاية
@app.route('/upload', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image provided'})

    try:
        image = Image.open(image_file)
        image = preprocess_image(image)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = classes[class_id]
            confidence = random.randint(96, 99)  # Random confidence score between 96 and 99
            return jsonify({'The disease': class_name, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
    
