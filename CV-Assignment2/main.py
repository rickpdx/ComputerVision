import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
import scikitplot as skplt

"""
Generates the base model and freezes it
Returns the model 
"""


def generateBaseModel():
    model = tf.keras.applications.InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3))

    # Freeze pre_model
    model.trainable = False

    return model


"""
Creates a new model with a transfer head
Params: a pre model
Returns: model
"""


def transferModel_1(pre_model):
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


"""
Evaluates the model's predictions
Converts the predictions to match the binary classes
Generates a confusion matrix jpg and prints out the confusion matrix
Params: Target labels, predicted labels
"""


def evaluate(labels, p):
    # Convert to 1 or 0
    y_pred = np.where(p >= 0.5, 1, 0)

    # Calculate and display accuracy
    acc = accuracy_score(labels, y_pred)
    print(f'Accuracy : {acc:.4f}')

    # Calculate and generate a figure of the confusion matrix
    cf = confusion_matrix(labels, y_pred)
    skplt.metrics.plot_confusion_matrix(
        labels, y_pred, title="Confusion Matrix", cmap=plt.cm.Reds)
    plt.savefig("confusionmatrix.jpg")

    # Display confusion matrix
    print(cf)


"""
Reads in images from a directory to create a dataset
Gets the labels from the dataset and normalizes the data
Params: directory string
Returns: inputs and labels
"""


def preprocess(src):
    # preprocess the images using keras method
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=src, color_mode="rgb", image_size=(150, 150), batch_size=50)

    # Convert data to numpy arrays, normalize data, and get labels
    counter = 0
    for x, y in ds:
        if counter == 0:
            labels = np.array(y)[:, None]
            norm = np.array(x)
            norm = np.round(norm/255, decimals=4)
            counter += 1
        else:
            xalt = np.array(x)
            xalt = np.round(xalt/255, decimals=4)
            yalt = np.array(y)[:, None]
            labels = np.concatenate((labels, yalt), axis=0)
            norm = np.concatenate((norm, xalt), axis=0)
    return norm, labels


"""
Plots the model structure and saves it to a pdf file. 
Displays the layer placement, name, and shape for analysis
"""


def plotModel(pre_model):

    # plot_model(pre_model, to_file='pre_model.pdf',
    #            show_shapes=True, show_layer_names=True)

    k = 0
    for layer in pre_model.layers:
        print("#", k, "-- name:", layer.name,
              "-- shape:", layer.output_shape)
        k += 1


def visualizeFilters(model):
    f = model.layers[1].get_weights()[0]
    if len(f.shape) == 4:
        f = np.squeeze(f)
        f = f.reshape((f.shape[0], f.shape[1], f.shape[2]*f.shape[3]))
        fig, axs = plt.subplots(5, 5, figsize=(8, 8))
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        for i in range(25):
            axs[i].imshow(f[:, :, i], cmap='gray')
            axs[i].set_title(str(i))
    plt.savefig(fname='filterVisualization.jpg')


def main():
    # Settings for threading
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

    # Datasets
    train_src = 'data/dataset/training_set/'
    test_src = 'data/dataset/test_set/'

    # Preprocess datasets and get labels
    trainX, trainY = preprocess(train_src)
    testX, testY = preprocess(test_src)

    # Instantiate the pre_model
    pre_model = generateBaseModel()

    # Visualizing first layer filters
    visualizeFilters(pre_model)

    # Analyze the pre_model
    # plotModel(pre_model)

    # Instantiate the subnet base model and freeze the weights
    sub_net = models.Model(pre_model.input, pre_model.layers[-186].output)
    sub_net.trainable = False

    # Analyze the sub_net
    # plotModel(sub_net)

    # Attach a transfer head
    model = transferModel_1(sub_net)
    plotModel(model)

    # Functions for the model
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam()
    metric = tf.keras.metrics.BinaryAccuracy()

    # Compile model
    # model.compile(metrics=[metric])
    model.compile(loss=loss_fn, optimizer=opt, metrics=[metric])

    # Fit the model
    model.fit(x=trainX, y=trainY, epochs=2)

    # Predict using batch size of 50
    p = model.predict(x=testX, batch_size=50)

    # Evaluate the predictions
    evaluate(testY, p)


if __name__ == "__main__":
    main()
