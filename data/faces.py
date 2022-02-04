import opendatasets as od

data_url = "https://www.kaggle.com/lamsimon/celebahq"

def download_faces_data():
    od.download(data_url)