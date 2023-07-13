# from flask import Flask, render_template, request, jsonify
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras.preprocessing.image as image_processing

# app = Flask(__name__)
# model = None

# def load_model():
#     global model
#     # Load your image classifier model here
#     model = tf.keras.models.load_model('dog_cat.h5')

# def preprocess_image(image):
#     # Preprocess the image before passing it to the model
#     image = image.resize((64, 64))  # Resize image to match model input shape
#     image = image_processing.img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     return image

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST','GET'])
# def predict():
#     if 'image' not in request.files:
#         return render_template('index.html', error='No image found.')
    
#     image = request.files['image']
#     image = Image.open(image).convert('RGB')
#     image = preprocess_image(image)

#     # Perform prediction
#     result=model.predict(image)

#     if result[0][0]==1:
#         pred='Dog'
#     else:
#         pred='Cat'
        
#     return render_template('index.html', class_name=pred)

# if __name__ == '__main__':
#     load_model()
#     app.run(debug=True,host='::',port=4000)
# 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from flask_cors import CORS
import numpy as np
import io
import cv2
from form import RegistrationForm, LoginForm

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
CORS(app)  # Enable CORS for all routes

model = models.mobilenet_v2(weights=None)
num_classes = 36  # Update with the number of classes in your model
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load('modelforclass.pth', map_location=torch.device('cpu')))
model.eval()


# Define the data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# class_labels = ['fapple', 'fbanana', 'fbittergroud', 'fcapsicum', 'fcucumber', 'fokra', 'forange', 'fpotato', 'ftomato',
#                 'rapple', 'rbanana', 'rbittergroud', 'rcapsicum', 'rcucumber', 'rokra', 'rorange', 'rpotato', 'rtomato']



class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
                'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi',
                'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate',
                'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                'watermelon']
@app.route('/')
def index():
    return redirect(url_for('register'))

@app.route("/register", methods=['GET', 'POST'])


@app.route('/Home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return render_template('index.html', error='No image found.')

    image = request.files['image']
    image_data = image.read()

    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]

    return render_template('index.html', class_name=predicted_label)


if __name__ == '__main__':
    app.run()
