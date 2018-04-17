import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2
from keras import optimizers
import datetime
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'resnet_fc_model.h5'
train_data_dir = '/home/cihan/Desktop/DATAFOLDER/train'
validation_data_dir = '/home/cihan/Desktop/DATAFOLDER/test'

# number of epochs to train top model
epochs = 200
# batch size used by flow_from_directory and predict_generator
batch_size = 16
bft_file = 'resnet_bottleneck_features_train.npy'
bfv_file = 'resnet_bottleneck_features_validation.npy'


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras


def save_bottleneck_features():
    # build the inception network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    bottleneck_features_train = model.predict_generator(generator, verbose=1)

    np.save(bft_file, bottleneck_features_train)
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, verbose=1)

    np.save(bfv_file,bottleneck_features_validation)


def train_top_model_bottleneck():
    start_time = datetime.datetime.utcnow()
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load(bft_file)

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_data = np.load(bfv_file)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=True, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=30, verbose=1, mode='auto')
    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[checkpoint, early])


    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    end_time = datetime.datetime.utcnow()

    print("Start time %s end time %s " % (str(start_time),str(end_time)))
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()




  


if __name__ == '__main__':
    from test_keras.tools.utils import evaluate_model
    # save_bottleneck_features()
    # train_top_model_bottleneck()

