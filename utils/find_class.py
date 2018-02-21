import ntpath
import os
import operator

def get_titles_from_folder(walk_dir,out_file):

    tuple_list = []
    total_value = 0
    value1=0
    sort_dict = {}
    for root, subdirs, files in os.walk(walk_dir):
        for file_name in files:
            head, tail = ntpath.split(root)
            total_value += len(files)
            item_name = (str(tail) + ":" + str(len(files)))
            sort_dict[tail] = len(files)
            tuple_list.append(item_name.lower())
            break

    sorted_x = sorted(sort_dict.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    tuple_list=sorted_x
    title_out = open(out_file, "w")
    title_out.write("Total item : %s\n" % total_value)
    title_out.write("Total class : %s\n\n" % len(tuple_list))
    for item in tuple_list:
        title_out.write("%s\n" % str(item))
    title_out.close()


def down_size_images(walk_dir):
    from PIL import Image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for root, subdirs, files in os.walk(walk_dir):
        for file_name in files:
            src_path = os.path.join(root,file_name)
            foo = Image.open(src_path)
            print("src size: %s" % str(foo.size))

            head, tail = ntpath.split(src_path)
            dst_dir = head.replace("ISIC","ISIC_RESIZE")
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_path = os.path.join(dst_dir,tail)
            foo = foo.resize((500, 500), Image.ANTIALIAS)
            foo.save(dst_path, quality=95,optimize=True)






if __name__ == '__main__':
    get_titles_from_folder("/home/cihan/Desktop/DATAFOLDER/test","test.txt")
    #down_size_images("/home/cihan/Downloads/ISIC")