import pandas as pd
from sklearn.model_selection import train_test_split
import config
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import PIL.Image as Image


class dataset(Dataset):

    def __init__(self, df, data_path, image_transform=None, train=True):  # Constructor.
        super(Dataset, self).__init__()  # Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)  # Returns the number of samples in the datasets.

    def __getitem__(self, index):
        image_id = self.df['id_code'][index]

        try:
            image = Image.open(f'{self.data_path}/{image_id}').convert('RGB')  # Image.
        except FileNotFoundError:
            print(FileNotFoundError)
            print(f'{self.data_path}/{image_id}')

        if self.image_transform:
            image = self.image_transform(image)  # Applies transformation to the image.

        if self.train:
            label = self.df['level'][index]  # Label.
            return image, label  # If train == True, return image & label.

        else:
            return image  # If train != True, return image.




def get_datasets():
    aptos = pd.read_csv(f'APTOS/train.csv')

    train_aptos, val_test_aptos = train_test_split(aptos, test_size=0.2, random_state=1)
    val_train_aptos, test_train_aptos = train_test_split(train_aptos, test_size=0.5, random_state=1)


    data_train = {
        "val_train_aptos"     : [train_aptos, f'APTOS/train'],
    }

    data_val = {
        "val_train_aptos"     : [val_train_aptos, f'APTOS/train'],
    }


    data_test = {
        "test_train_aptos"     : [test_train_aptos, f'APTOS/train'],
    }



    return data_train, data_val, data_test




