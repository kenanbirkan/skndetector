from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import shutil
import os

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

img = load_img('/home/user/Desktop/TEST_KERAS/2600_0_3549.jpeg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


save_path = '/home/user/Desktop/TEST_KERAS/preview'
shutil.rmtree(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,save_to_dir=save_path, save_prefix='2600', save_format='jpeg'):
    print(1)
    exit(1)