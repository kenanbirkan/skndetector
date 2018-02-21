import json
import os
import traceback
import shutil

dermnet_dir = "/media/cihan/windows/DERMIS/"
download_dir = "/media/cihan/windows/DERMIS/"

def label_image():

    with open('dermis_output.json') as data_file:
        data = json.load(data_file)

    for elem in data:
        try:
            class_label = elem["title"].split("diagnosis:")[1].replace("\n","").replace("\t","")
            print class_label
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


def get_titles():
    from collections import Counter
    with open('dermis_output.json') as data_file:
        data = json.load(data_file)

    title_list = []
    for elem in data:
        try:
            class_label = elem["title"].split("diagnosis:")[1].replace("\n", "").replace("\t", "")
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
    title_out = open("dermis_title.txt","w")
    title_out.write("Total item : %s\n" % total_value)
    title_out.write("Total class : %s\n" % len(tuple_list))
    for item in tuple_list:
        title_out.write("%s\n" % item)
    title_out.close()



if __name__ == '__main__':
    get_titles()







