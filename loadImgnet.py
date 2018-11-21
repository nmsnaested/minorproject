from PIL import Image
import requests
from io import BytesIO

class GetDataset():
    def __init__(self, dirname, transforms=None): 
        self.dirname = dirname
        self.transforms = transforms

    def __getitem__(self, idx): 

    def __len__(self):
        return len(self.list)


        
response = requests.get(url)
img = Image.open(BytesIO(response.content))