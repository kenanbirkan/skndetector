from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os

# dimensions of our images.
img_width, img_height = 150, 150

def train_base_model(train_data_dir,validation_data_dir,epochs,batch_size):


    nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
    nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])



    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = create_model(input_shape)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    train_label_map = (train_generator.class_indices)
    print("train_label_map:%s " % str(train_label_map))

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        workers = 3)

    model.save('first_try_all.h5')


def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model


def predict_input(model_name,input_image):
    from keras.models import load_model
    from keras.preprocessing.image import  img_to_array, load_img

    import cv2
    import numpy as np

    # if K.image_data_format() == 'channels_first':
    #     input_shape = (3, img_width, img_height)
    # else:
    #     input_shape = (img_width, img_height, 3)
    #
    # model = create_model(input_shape=input_shape)
    #
    # model.load_weights(model_name)
    #
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    model = load_model(model_name)

    img = cv2.imread(input_image)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict_proba(img)



    print prediction



if __name__ == '__main__':
    # HOME
    # train_data_dir = '/home/cihan/Desktop/DATAFOLDER/train'
    # validation_data_dir = '/home/cihan/Desktop/DATAFOLDER/test'
    # epochs = 20
    # batch_size = 16
    # train_base_model(train_data_dir=train_data_dir,validation_data_dir=validation_data_dir,epochs=epochs,batch_size=batch_size)
    #
    #WORK

    # train_data_dir = '/home/user/Desktop/DATAFOLDER/train'
    # validation_data_dir = '/home/user/Desktop/DATAFOLDER/test'
    # epochs = 20
    # batch_size = 16
    # train_base_model(train_data_dir=train_data_dir, validation_data_dir=validation_data_dir, epochs=epochs, batch_size=batch_size)

    # predict
    predict_input("first_try_all.h5","/home/cihan/Desktop/DATASET/rosacea/0d34af9db56295f741134c905101b5bbf04ea08a.JPG")