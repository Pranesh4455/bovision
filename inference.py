import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("bovision_ai_classifier.keras")

class_names = ["buffalo", "cow"]

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])

    print(f"Prediction: {class_names[class_index]}")
    print(f"Confidence: {np.max(predictions[0])*100:.2f}%")

# Example
predict("test.jpg")
