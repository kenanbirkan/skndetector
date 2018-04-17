import gc
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import *
from keras import applications as apps
from keras import optimizers as opt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
train_data_dir = '/home/cihan/Desktop/DATAFOLDER/train'
validation_data_dir = '/home/cihan/Desktop/DATAFOLDER/test'
# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16

def setup_to_transfer_learn(model, base_model):
    """Setup the models for transfer learning"""
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def setup_to_finetune(model, n):
    """Setup the models for finetunning."""
    # Setting everything bellow n to be not trainable
    for i, layer in enumerate(model.layers):
        layer.trainable = i > n

    model.compile(
        optimizer=opt.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    filename_model_json = 'model.json'
    filename_model_weights = 'model.h5'
    batch_size = 16
    num_classes = 4


    base_model = apps.resnet50.ResNet50(include_top=False)
    ppf = apps.resnet50.preprocess_input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=pred)

    train_gen = ImageDataGenerator(
        # preprocessing_function=ppf,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = train_gen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size
    )

    test_gen = ImageDataGenerator(
        # preprocessing_function=ppf
        # rescale=1./255
    )

    test_gen = test_gen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size
    )

    model = setup_to_transfer_learn(model, base_model)

    model.fit_generator(
        train_gen,
        epochs=2,
        validation_data=test_gen,
        class_weight='auto'
    )

    model = setup_to_finetune(model, len(base_model.layers) - 1)

    chkpt_save_best = ModelCheckpoint(filename_model_weights, monitor='val_acc',
                                      verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')
    history = model.fit_generator(
        train_gen,
        epochs=100,
        validation_data=test_gen,
        class_weight='auto',
        callbacks=[chkpt_save_best,early]
    )


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

    # Saving the model structure and its weights
    model_json = model.to_json()
    with open(filename_model_json, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename_model_weights)

    print(model.evaluate_generator(test_gen, 10, workers=4))


if __name__ == '__main__':
    main()