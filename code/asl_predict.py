import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse

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

def predict_letter(file_path, model):   # classify_image
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)

    prediction_probability = prediction[0, prediction.argmax(axis=1)][0]

    # convert prediction to letter
    predicted_letter = alphabet[np.argmax(prediction)]

    print("[Info] Predicted: {}, Confidence: {}".format(predicted_letter, 
                                                        prediction_probability))
    return predicted_letter

def main():
    # Setup the argument parser to parse out command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--augument", type=int, default=1,
                    help="(optional) Which trained model (CNN or Augmented) should be used for predict/classify image. Default is augumented.")

    args = vars(ap.parse_args())
    if args["augument"] > 0:
        model = keras.models.load_model("data/asl_augment_model.h5")
    else:
        model = keras.models.load_model("data/asl_cnn_model.h5")
    
    # model.summary()

    # https://www.startasl.com/fingerspelling/
    predict_letter("data/asl_images/a.png", model)
    predict_letter("data/asl_images/b_1.png", model)
    predict_letter("data/asl_images/y.png", model)
    predict_letter("data/asl_images/f.png", model)
    predict_letter("data/asl_images/e.png", model)

if __name__ == "__main__":
    # The American Sign Language Letters dataset is an object detection dataset of 
    # each ASL letter with a bounding box.
    main()