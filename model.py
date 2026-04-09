import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

print("Loading model...")

model = tf.keras.models.load_model("agriculture_model.keras")

class_names = ['disease', 'healthy', 'pest']

print("Model Loaded Successfully")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    return predicted_class, round(confidence, 2)