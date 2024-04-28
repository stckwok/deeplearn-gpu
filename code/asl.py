import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import argparse

# imports used to build the deep learning model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def show_images(num_images, x_train, y_train):
    num_images = 10 # 20
    for i in range(num_images):
        row = x_train[i]
        label = y_train[i]
        
        image = row.reshape(28,28)
        plt.subplot(1, num_images, i+1)
        plt.title(label, fontdict={'fontsize': 30})
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.show()

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

def build_simple_model(num_classes, num_units=512, shape_num=784):
    '''
    Build a sequential model that has
     - a dense input layer containing 512 neurons, use the `relu` activation function, 
        and expect input images with a shape of `(784,)`
     - second dense layer with 512 neurons which uses the `relu` activation function
     - dense output layer with neurons equal to the number of classes, using the `softmax` activation function

    '''
    # define a function to build the model
    model = Sequential()
    model.add(Dense(units = num_units, activation='relu', input_shape=(shape_num,)))
    model.add(Dense(units = num_units, activation='relu'))
    model.add(Dense(units = num_classes, activation='softmax'))

    model.summary()
    return model

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

def main():
    # Setup the argument parser to parse out command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train-cnn", type=int, default=1,
                    help="(optional) Which model (CNN or not) should be used for training with the ASL dataset. Defaults to yes")
    ap.add_argument("-n", "--epocs", type=int, default=20,
                    help="(optional) Number of epocs used for training. Defaults to 20")
    args = vars(ap.parse_args())

    print("[INFO] Loading the ASL dataset...")
    # dataset is available from the website [Kaggle](http://www.kaggle.com)
    # Read the CSV files into a format called a [DataFrame] from Pandas
    # See (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).
    train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
    valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")
    print(train_df.head())  # print first few rows 

    # extract the labels
    y_train = train_df['label']
    y_valid = valid_df['label']
    del train_df['label']
    del valid_df['label']
    # store training and validation images in `x_train` and `x_valid` variables.
    x_train = train_df.values
    x_valid = valid_df.values

    # individual pictures in dataset are in the format of long lists of 784 pixels.
    # We don't have all the information about which pixels are near each other. 
    # Because of this, we can't apply convolutions that will detect features.

    #  27,455 images with 784 pixels each for training...
    print("x train : ", x_train.shape)
    print("y train (corresponding labels): ", y_train.shape)

    # For validation, we have 7,172 images...
    print("x valid : ", x_valid.shape) 
    print("y valid (corresponding labels): ", y_valid.shape)

    # show_images(10, x_train, y_train)

    # Rescale the data from values between [0 - 255] to [0 - 1.0]
    x_train = train_df.values / 255
    x_valid = valid_df.values / 255

    # Two letters (j and z) require movement, so they are not included in the training dataset.
    num_classes = 24
    # Turn our scalar targets into binary categories
    if not y_train.shape[-1] == 24:  # Avoid running multiple times
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)

    # The data is all prepared now, we have normalized images for training and validation, 
    # as well as categorically encoded labels for training and validation

    # if CNN:
    if args["train_cnn"] > 0:
        # Reshape dataset to be in a 28x28 pixel format. 
        # This allows our convolutions to associate groups of pixels and detect important features.
        print("[INFO] CNN - Reshape data before building CNN model...\n")
        x_train = x_train.reshape(-1,28,28,1)
        x_valid = x_valid.reshape(-1,28,28,1)
        model=build_cnn_model(num_classes)
        opt = SGD(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', 
                    optimizer=opt, metrics=['accuracy'])
    else:
        print("[INFO] building Simple model...\n")
        model = build_simple_model(num_classes)
        model.compile(loss='categorical_crossentropy', 
                    metrics=['accuracy'])

    # Train the model
    print("[INFO] Training the model...")
    history = model.fit(x_train, y_train, 
                        epochs=args["epocs"], #20, 
                        verbose=1, 
                        validation_data=(x_valid, y_valid))

    # Use the test data to evaluate the model
    print("[INFO] Evaluating the model...")
    (loss, accuracy) = model.evaluate(
        x_valid, y_valid, batch_size=128, verbose=1)

    print("[INFO] Accuracy: {:.2f}%".format(accuracy * 100))

    print("[INFO] Saving the model to a h5 file...")

    # if CNN:
    if args["train_cnn"] > 0:
        # Visualize the training history
        graph_training_history(history, title="Using CNN")
        # model.save_weights("data/asl_weights.hdf5", overwrite=True)
        model.save("data/asl_cnn_model.h5")
    else:
        # Visualize the training history
        graph_training_history(history, title="Without CNN")
        # model.save_weights("data/asl_weights.hdf5", overwrite=True)
        model.save("data/asl_simple_model.h5")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    # The American Sign Language Letters dataset is an object detection dataset of each ASL letter with a bounding box.
    main()