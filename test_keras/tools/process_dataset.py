import ntpath
import os
import operator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil
import numpy as np

def get_max_counts(walk_dir):
    sort_dict = {}
    for root, subdirs, files in os.walk(walk_dir):
        for file_name in files:
            head, tail = ntpath.split(root)
            sort_dict[tail] = len(files)
            break
    sorted_x = sorted(sort_dict.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    return sorted_x


def augment_data(current_index, image_path, save_path, n_count):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    img = load_img(image_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    if current_index < n_count:
        current_index += 1
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpeg'):
            print(current_index)
            return current_index
    else:
        return -1


def balance_dir(sorted_list, walk_dir):
    max_count = sorted_list[0][1]
    for item in sorted_list:
        current_index = 0
        current_dir = os.path.join(walk_dir, item[0])
        n_count = max_count - item[1]
        if n_count:
            is_finished = False
            while not is_finished:
                for file_name in os.listdir(current_dir):
                    if "aug" not in file_name:
                        current_index = augment_data(current_index=current_index, image_path=os.path.join(current_dir, file_name), save_path=current_dir, n_count=n_count)
                        if current_index == -1:
                            is_finished=True
                            break


def start_balance_dataset(walk_dir):
    sorted_list = get_max_counts(walk_dir)
    balance_dir(sorted_list=sorted_list, walk_dir=walk_dir)


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


if __name__ == '__main__':
    walk_dir = "/home/cihan/Desktop/DATASET"
    start_balance_dataset(walk_dir)
    split_dataset_into_test_and_train_sets(all_data_dir=walk_dir,
                                           testing_data_dir="/home/cihan/Desktop/DATAFOLDER/test",
                                           training_data_dir="/home/cihan/Desktop/DATAFOLDER/train",
                                           testing_data_pct=0.2)

