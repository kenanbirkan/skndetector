import datetime
import os

import numpy as np
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import ntpath

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'fc_model.h5'
train_data_dir = '/home/cihan/Desktop/DATAFOLDER/train'
validation_data_dir = '/home/cihan/Desktop/DATAFOLDER/test'

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def train_top_model():
    start_time = datetime.datetime.utcnow()
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    #model.add(top_model)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    nb_train_samples = len(train_generator.filenames)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    nb_validation_samples =len(validation_generator.filenames)

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)


    end_time = datetime.datetime.utcnow()

    print("Start time %s end time %s " % (str(start_time), str(end_time)))

    model.save("keras4_100epoch.h5")

    score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])

def predict(image_path,model=None):

    if not model:
        model = load_model("keras4_100epoch.h5")

    class_dictionary = np.load('class_indices.npy').item()
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    class_predicted =np.argmax(prediction, axis=1)
    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))
    return label

def predict_folder_path(walk_dir):
    result_list = []
    model = load_model("keras4.h5")

    for root, subdirs, files in os.walk(walk_dir):
        positive = 0
        negative = 0
        result_dict = {}
        true_label = None
        for file_name in files:
            head, true_label = ntpath.split(root)
            full_path = os.path.join(root,file_name)
            label = predict(full_path,model=model)
            if label == true_label:
                positive+=1
            else:
                negative+=1

        if true_label:
            result_dict[true_label+":positive"] = positive
            result_dict[true_label + ":negative"] = negative
            result_list.append(result_dict)

    print result_list







if __name__ == '__main__':

    #train_top_model()
    # predict("/home/cihan/Desktop/DATAFOLDER/test/rosacea/8f59caa743b399909ffe38ddb918c2ab6fbaffd9.jpg")
    predict_folder_path("/home/cihan/Desktop/DATAFOLDER/test")