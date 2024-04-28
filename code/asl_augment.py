import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import argparse

# imports used to build the deep learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# function to graph the training history of the model
def graph_training_history(history, title):
    plt.rcParams["figure.figsize"] = (12, 9)
    plt.style.use('ggplot')
    plt.figure(title)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

def build_cnn_model(num_classes):
    from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,)

    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                    input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.summary()
    return model

def show_image_batch(img_iter, batch_size):
    x, y = img_iter.next()
    fig, ax = plt.subplots(nrows=4, ncols=8)
    for i in range(batch_size):
        image = x[i]
        ax.flatten()[i].imshow(np.squeeze(image))
    plt.show()

def main():
    # Load in our data from CSV files
    train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
    valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")

    # 1. Data Preparation
    # Separate out our target values
    y_train = train_df['label']
    y_valid = valid_df['label']
    del train_df['label']
    del valid_df['label']

    # Separate our our image vectors
    x_train = train_df.values
    x_valid = valid_df.values

    # Turn our scalar targets into binary categories
    num_classes = 24
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    # Normalize our image data
    x_train = x_train / 255
    x_valid = x_valid / 255

    # Reshape the image data for the convolutional network
    x_train = x_train.reshape(-1,28,28,1)
    x_valid = x_valid.reshape(-1,28,28,1)

    # 2. Model creation
    model=build_cnn_model(num_classes)

    # 3. Data Augmentation
    # In order to teach our model to be more robust when looking at new data, we're going to 
    # programmatically increase the size and variance in our dataset.
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False, # Don't randomly flip images vertically
    )  

    # The number of examples from the training dataset used in the estimate of the error gradient 
    # is called the batch size and is an important hyperparameter that influences the dynamics 
    # of the learning algorithm.
    batch_size = 32
    img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)
    show_image_batch(img_iter, batch_size)
    
    # 4. Fit data to Generator
    datagen.fit(x_train)

    # 5. Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # 6. Training with Augumentation
    history= model.fit(img_iter,
                    epochs=10,
                    # Run same number of steps we would if we were not using a generator.
                    steps_per_epoch=len(x_train)/batch_size, 
                    validation_data=(x_valid, y_valid))

    graph_training_history(history, title="Using Data Augmentation")
    # Note: validation accuracy is higher, and more consistent. 
    # The Model is no longer overfitting in the way it was; it generalizes better, 
    # making better predictions on new data.

    # Use the test data to evaluate the model
    print("[INFO] Evaluating the model...")
    (loss, accuracy) = model.evaluate(
        x_valid, y_valid, batch_size=128, verbose=1)

    print("[INFO] Accuracy: {:.2f}%".format(accuracy * 100))

    print("[INFO] Saving the augmented model to a h5 file...")
    # 7. Save model (i.e. weights)
    # A well-trained model is saved to disk. Next step is to 
    # deploy the model and make predictions on not-yet-seen images.
    model.save('data/asl_augment_model.h5')
    model.summary()


if __name__ == "__main__":
    # The American Sign Language Letters dataset is an object detection dataset of 
    # each ASL letter with a bounding box.
    main()
