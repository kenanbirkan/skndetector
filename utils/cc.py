import requests

r = requests.get("https://isic-archive.com/api/v1/image/5436e3cbbae478396759f297/download")
if r.status_code == 200:
    with open("/home/user/Desktop/TEST_KERAS/cc.jpg", 'wb') as f:
        f.write(r.content)