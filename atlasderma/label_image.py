import json
import os
import traceback
import shutil

dermnet_dir = "/media/cihan/windows/ATLASDERMA/"
download_dir = "/media/cihan/windows/ATLASDERMA/"


def label_image():
    with open('atlas_derma_output.json') as data_file:
        data = json.load(data_file)

    for elem in data:
        try:
            class_label = elem["title"].split("(")[0]
            print (class_label)
            image_path = elem["files"][0]["path"]
            folder_path = dermnet_dir + class_label
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            src_path = download_dir + image_path
            a, image_name = image_path.split("/")
            dst_path = dermnet_dir + class_label + "/" + image_name
            print ("moving src: %s to dst : %s  " % (src_path, dst_path))
            shutil.copy(src_path, dst_path)


        except:
            print(traceback.format_exc())


def get_titles():
    from collections import Counter
    with open('atlas_derma_output.json') as data_file:
        data = json.load(data_file)

    title_list = []
    for elem in data:
        try:
            class_label = elem["title"].split("(")[0]
            title_list.append(class_label)
        except:
            print(traceback.format_exc())
    counter = dict(Counter(title_list))

    tuple_list = []
    total_value = 0
    for key, value in counter.iteritems():
        try:
            total_value += value
            str_val = u':'.join((key, str(value))).encode('utf-8').strip()
            tuple_list.append(str(str_val).lower())
        except:
            print(traceback.format_exc())

    tuple_list.sort()
    title_out = open("altas_derma_title.txt","w")
    title_out.write("Total item : %s\n" % total_value)
    title_out.write("Total class : %s\n" % len(tuple_list))
    for item in tuple_list:
        title_out.write("%s\n" % item)
    title_out.close()



def get_titles_from_folder(walk_dir):
    import ntpath
    tuple_list = []
    total_value = 0
    for root, subdirs, files in os.walk(walk_dir):
        for file_name in files:
            head, tail = ntpath.split(root)
            total_value += len(files)
            item_name = (str(tail) + ":" + str(len(files)))
            tuple_list.append(item_name.lower())
            break

    tuple_list.sort()
    title_out = open("test.txt", "w")
    title_out.write("Total item : %s\n" % total_value)
    title_out.write("Total class : %s\n\n" % len(tuple_list))
    for item in tuple_list:
        title_out.write("%s\n" % item)
    title_out.close()

if __name__ == '__main__':
    get_titles_from_folder("/home/user/Desktop/test_scrapy")
