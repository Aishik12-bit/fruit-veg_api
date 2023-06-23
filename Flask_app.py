import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from flask_cors import CORS

app= Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('cnn_model')

# Use the loaded model for predictions or further training
classes_list = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
                'cilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
                'kiwi', 'lemon', 'lettuc',  'mango', 'onion', 'orange',  'paprika', 
                'pear', 'peas', 'pineapple','pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
                'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

@app.route('/test', methods=['GET', 'POST'])
def output():
    print("hello")
    return "hello"


@app.route('/identify', methods=['POST'])
def identify():
    if 'image' not in request.files:
        return jsonify({'error': 'Image not found'})
    img = request.files['image']
    img_path = 'image.jpg'  # Specify the path to save the uploaded image
    img.save(img_path)  # Save the image file

    test_img = image.load_img(img_path, target_size=(256, 256))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    predicted_class_index = np.argmax(result[0])
    classification = classes_list[predicted_class_index]
    
    return jsonify({'classification': classification})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




#print("Predicted Class: ", pred)
