
from flask import Flask, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path
import base64
import uuid

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

# resolve compatibility problems in TensorFlow, cuDNN, and Flask
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# dimensions of our images.
img_width, img_height = 224, 224
# limiting the allowed filetypes
ALLOWED_FILETYPES = set(['.jpg', '.jpeg', '.gif', '.png'])

model_path = 'models/asl_augment_model.h5'

# loading the trained model
model = load_model(model_path)

# Alphabet does not contain j or z because they require movement
alphabet = "abcdefghiklmnopqrstuvwxy"

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
    return image

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')

def predict_letter(file_path, model):
    # show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1) 
    image = image/255
    prediction = model.predict(image)

    prediction_probability = prediction[0, prediction.argmax(axis=1)][0]

    # convert prediction to letter    
    label = alphabet[np.argmax(prediction)]

    print("[Info] Predicted: {}, Confidence: {}".format(label, 
                                                        prediction_probability))
    return label, prediction_probability

# get a thumbnail version of the uploaded image
def get_img_thumbnail(image):
    image.thumbnail((400, 400), resample=Image.LANCZOS)
    image = image.convert("RGB")
    with BytesIO() as buffer:
        image.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

# request handler function for the home/index page
def index():
    # handling the POST method of the submit
    if request.method == 'POST':
        # check if the post request has the submitted file
        if 'letter_image' not in request.files:
            print("[Error] No file uploaded.")
            flash('No file uploaded.')
            return redirect(url_for('index'))
        
        f = request.files['letter_image']
        print("\nFilename : ", f.filename)
        # if no file is selected by the user, some browsers may
        # submit an empty field without the filename
        if f.filename == '':
            print("[Error] No file selected to upload.")
            flash('No file selected to upload.')
            return redirect(url_for('index'))

        sec_filename = secure_filename(f.filename)
        file_extension = os.path.splitext(sec_filename)[1]

        if f and file_extension.lower() in ALLOWED_FILETYPES:
            file_tempname = uuid.uuid4().hex
            image_path = './uploads/' + file_tempname + file_extension
            f.save(image_path)

            label, prediction_probability = predict_letter(image_path, model)
            prediction_probability = np.around(prediction_probability * 100, decimals=4)

            orig_image = Image.open(image_path)
            image_data = get_img_thumbnail(image=orig_image)

            with application.app_context():
                return render_template('index.html', 
                                        label=label, 
                                        prob=prediction_probability,
                                        image=image_data
                                        )
        else:
            print("[Error] Unauthorized file extension: {}".format(file_extension))
            flash("The file type you selected: '{}' is not supported. Please select a '.jpg', '.jpeg', '.gif', or a '.png' file.".format(file_extension))
            return redirect(url_for('index'))
    else:
        # handling the GET, HEAD, and any other methods
        # 
        with application.app_context():
            return render_template('index.html')

# handle 'filesize too large' errors
def http_413(e):
    print("[Error] Uploaded file too large.")
    flash('Uploaded file too large.')
    return redirect(url_for('index'))

# setting up the application context
application = Flask(__name__)
# set the application secret key. Used with sessions.
application.secret_key = '@#$%^&*@#$%^&*$%$%$##@#$'

# add a rule for the index page.
# This approach for URL mapping when we are importing the view function from another module.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

# limit the size of the uploads
application.register_error_handler(413, http_413)
application.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. 
    # This line should be removed before deploying a production app.
    application.debug = True
    application.run(port=8002)