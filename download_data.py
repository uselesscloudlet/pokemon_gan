import os
import zipfile
from configparser import ConfigParser

import wget

config = ConfigParser()
config.read('conf/download_data.conf')
url = config['DOWNLOAD']['Data_url']

wget.download(url)

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('')

os.remove('archive.zip')
