import requests
import json
import traceback
import shutil
import os


def download_data(download_path):
    count = 0
    for i in range(106, 200):
        link = "https://isic-archive.com/api/v1/image?limit=100&sort=name&sortdir=1&offset=%s&detail=true" % (i*100)
        print("Starting request %s" % link)
        r = requests.get(link)
        print("Finished request %s" % link)
        if r.status_code == 200:
            try:
                json_response = json.loads(r.text)
            except:
                continue
            for item in json_response:
                try:
                    count += 1
                    image_type = item["meta"]["acquisition"]["image_type"]
                    diagnosis = item["meta"]["clinical"]["diagnosis"]
                    benign_malignant = item["meta"]["clinical"]["benign_malignant"]
                    id = item["_id"]
                    link = "https://isic-archive.com/api/v1/image/%s/download" % id
                    print("Starting request %s" % link)
                    try:
                        r2 = requests.get(link,timeout=(100,100))
                    except:
                        print("Tiemout occured")
                        continue
                    print("Finished request %s" % link)
                    if r2.status_code == 200:
                        image_dir_path = os.path.join(download_path, image_type, diagnosis + "-" + benign_malignant)
                        if not os.path.exists(image_dir_path):
                            os.makedirs(image_dir_path)
                        with open(os.path.join(image_dir_path, str(id) + ".jpg"), 'wb') as out_file:
                            out_file.write(r2.content)
                        print("count :%s diagnosis:%s  benign_malignant:%s id%s " % (count, diagnosis, benign_malignant, id))
                except:
                    print traceback.format_exc()


if __name__ == '__main__':
    download_data("/home/cihan/Downloads/ISIC/")
