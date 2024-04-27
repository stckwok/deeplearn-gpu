import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time

from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils

# Alphabet does not contain j or z because they require movement
alphabet = "abcdefghiklmnopqrstuvwxy"

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')

def predict_letter(file_path):
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)
    # convert prediction to letter
    predicted_letter = alphabet[np.argmax(prediction)]
    return predicted_letter

model = keras.models.load_model("data/asl_cnn_model.h5")
# model.summary()

# https://www.startasl.com/fingerspelling/
print(predict_letter("data/asl_images/a.png"))
print(predict_letter("data/asl_images/a_1.png"))
print(predict_letter("data/asl_images/b_1.png"))
print(predict_letter("data/asl_images/a.png"))
print(predict_letter("data/asl_images/a.png"))
print(predict_letter("data/asl_images/c.png"))