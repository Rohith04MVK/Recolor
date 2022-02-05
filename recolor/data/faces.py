import os

import opendatasets as od

data_url = "https://www.kaggle.com/lamsimon/celebahq"

def download_faces_data(save_path="~/datasets/celebahq"):
    save_path = os.path.expanduser(save_path)
    try:
        os.path.exists(save_path)
    except:
        od.download(data_url, data_dir=save_path)