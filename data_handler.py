import pandas as pd
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        # A.augmentations.transforms.HueSaturationValue(),
        # A.augmentations.transforms.RGBShift(),
        # A.augmentations.transforms.RandomBrightness(),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def get_df(type, config):
    if type == 'train':
        csv_file = config.train_dir
    elif type == 'test':
        csv_file = config.test_dir
    else:
        print('No such type: {}'.format(type))

    df = pd.read_csv(f'{config.data_dirs[type]}.csv')

    if type == 'train':
        df['x'] = -1
        df['y'] = -1
        df['w'] = -1
        df['h'] = -1
        df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
        df.drop(columns=['bbox'], inplace=True)
        df['x'] = df['x'].astype(np.float)
        df['y'] = df['y'].astype(np.float)
        df['w'] = df['w'].astype(np.float)
        df['h'] = df['h'].astype(np.float)
    return df

def get_ds(config):
    train_df = get_df('train', config) #getting the DataFrame
    #split the images from train to train and validation
    image_ids = train_df['image_id'].unique()
    np.random.seed(0)
    image_ids=pd.Series(image_ids).sample(frac=1).values
    train_size = int(config.train_ratio * config.num_of_pics_to_use)
    valid_size = int(config.valid_ratio * config.num_of_pics_to_use)

    
    train_ids = image_ids[:train_size]
    valid_ids = image_ids[train_size:train_size + valid_size]
    #creating the train and validation df - including the relevant bboxes
    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]
    
    #create dataset instnce for train and validation seperatly
    train_dataset = WheatDataset(train_df, config.data_dirs['train'], get_train_transform())
    valid_dataset = WheatDataset(valid_df, config.data_dirs['train'], get_valid_transform())

    return train_dataset, valid_dataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(config):
    train_dataset, valid_dataset = get_ds(config)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,  # The num of workers defines the number of threads (0 - only main thread is working).
        # i currently set it to 0 until we learn how to work with it
        collate_fn=collate_fn
    )

    return train_data_loader, valid_data_loader