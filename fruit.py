import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('cnn_model')

# Use the loaded model for predictions or further training
classes_list = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'eggplant', 
                'peas', 'orange', 'pomegranate', 'pear', 'grapes', 'cauliflower', 'raddish', 'pineapple', 
                'lettuce', 'corn', 'soy beans',  'paprika', 'sweetpotato', 'lemon',  'onion', 
                'watermelon', 'potato', 'kiwi', 'ginger', 'chilli pepper', 'jalepeno', 'garlic', 
                'tomato', 'turnip', 'mango', 'sweetcorn', 'spinach', 'cucumber']
@app.route('/test', methods=['POST'])
def output():
    print("hello")
@app.route('/identify', methods=['POST'])
def identify():
    if 'image' not in request.files:
        return jsonify({'error':'Image not found'})
    # img=request.files['image'] 
    # test_img = image.load_img(img, target_size=(256, 256))
    # test_img = image.img_to_array(test_img)
    # test_img = np.expand_dims(test_img, axis=0)
    # result = model.predict(test_img)  # Use the loaded model as a function to make predictions
    # predicted_class_index = np.argmax(result[0])  # Get the index of the class with the highest probability 
    # jsonify({'classification': classes_list[predicted_class_index]})

if __name__ == '__main__':
    app.run()




#print("Predicted Class: ", pred)                