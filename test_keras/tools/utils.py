import ntpath
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from test_keras.vgg16.vgg16_bottleneck import validation_data_dir, img_height, img_width

# number of epochs to train top model
epochs = 100  # TODO update
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def predict(image_path, model=None, model_path=None):
    if not model:
        model = load_model(model_path)

    class_dictionary = np.load('class_indices.npy').item()
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    class_predicted = np.argmax(prediction, axis=1)
    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))
    return label


def predict_folder_path(walk_dir, model_path):
    result_list = []
    model = load_model(model_path)

    for root, subdirs, files in os.walk(walk_dir):
        positive = 0
        negative = 0
        result_dict = {}
        true_label = None
        for file_name in files:
            head, true_label = ntpath.split(root)
            full_path = os.path.join(root, file_name)
            label = predict(full_path, model=model)
            if label == true_label:
                positive += 1
            else:
                negative += 1

        if true_label:
            result_dict[true_label + ":positive"] = positive
            result_dict[true_label + ":negative"] = negative
            result_list.append(result_dict)

    print result_list

def evaluate_model_resnet(model_path):
    from keras.applications.resnet50 import preprocess_input
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    nb_validation_samples = len(validation_generator.filenames)
    score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])


def evaluate_model(model_path):
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    nb_validation_samples = len(validation_generator.filenames)
    score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])

def evaluate_model_inc(model_path):
    from keras.applications.inception_v3 import preprocess_input
    model = load_model(model_path)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical')
    nb_validation_samples = len(validation_generator.filenames)
    score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])