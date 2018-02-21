import json
import os
import traceback
import shutil

dermnet_dir = "/media/cihan/windows/HELLENIC/"
download_dir = "/media/cihan/windows/HELLENIC/"

def label_image():

    with open('output.json') as data_file:
        data = json.load(data_file)

    for elem in data:
        try:
            class_label = elem["title"]
            image_path = elem["files"][0]["path"]
            folder_path = dermnet_dir + class_label
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            src_path =download_dir + image_path
            a,image_name = image_path.split("/")
            dst_path = dermnet_dir + class_label + "/" + image_name
            print ("moving src: %s to dst : %s  " % (src_path,dst_path) )
            shutil.copy(src_path,dst_path)


        except:
            print(traceback.format_exc())


def count_images():
    with open('/media/cihan/windows/DERMNET/file_counts.txt') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            array = line.split('file')
            numb = array[0].replace(' ', '')
            print numb
            count += int(numb)
        print count


if __name__ == '__main__':
    label_image()







