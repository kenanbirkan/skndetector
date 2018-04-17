import datetime

import matplotlib.pyplot as plt
from resnet_bottleneck import train_top_model_bottleneck,top_model_weights_path, train_data_dir, validation_data_dir, img_height, img_width
from keras import applications
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# number of epochs to train top model
epochs = 100
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def train_top_model():
    start_time = datetime.datetime.utcnow()
    # build the VGG16 network
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    print('Model loaded.')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    # model.add(top_model)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    for layer in model.layers[:79]:
        layer.trainable = False
    for layer in model.layers[79:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    nb_validation_samples = len(validation_generator.filenames)

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint("/home/cihan/Desktop/resnet50.h5", monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

    # fine-tune the model
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        shuffle=True,
        callbacks=[checkpoint, early])

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

    end_time = datetime.datetime.utcnow()

    print("Start time %s end time %s " % (str(start_time), str(end_time)))

    score = model.evaluate_generator(validation_generator, nb_validation_samples / batch_size)
    print("Loss: ", score[0], "Accuracy: ", score[1])


if __name__ == '__main__':
    from test_keras.tools.utils import evaluate_model_resnet

    train_top_model_bottleneck()
    train_top_model()

    evaluate_model_resnet("/home/cihan/Desktop/resnet50.h5")
