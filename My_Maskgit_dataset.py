import os
import random
import shutil
import sys
import numpy as np
import time
from pathlib import Path
from threading import Thread
import datasets

import torch

from datasets import Image, load_from_disk
from PIL import (
    Image as pImage,
    ImageFile,
)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from io import BytesIO
import requests
#import MAX_LENGTH
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    from tqdm_loggable.auto import tqdm
except ImportError:
    from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_size,
        image_column="image",
        flip=True,
        center_crop=True,
        stream=False,
        using_taming=False,
        random_crop=False,
        alpha_channel=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.stream = stream
        transform_list = [
            T.Resize(image_size),
        ]

        if flip:
            transform_list.append(T.RandomHorizontalFlip())
        if center_crop and not random_crop:
            transform_list.append(T.CenterCrop(image_size))
        if random_crop:
            transform_list.append(T.RandomCrop(image_size, pad_if_needed=True))
        if alpha_channel:
            transform_list.append(T.Lambda(lambda img: img.convert("RGBA") if img.mode != "RGBA" else img))
        else:
            transform_list.append(T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img))

        transform_list.append(T.ToTensor())
        self.transform = T.Compose(transform_list)
        self.using_taming = using_taming

    def __len__(self):
        if not self.stream:
            return len(self.dataset)
        else:
            raise AssertionError("Streaming doesnt support grabbing dataset length")

    def __getitem__(self, index):
        try:
            image = self.dataset[index][self.image_column]
            if self.using_taming:
                return self.transform(image) - 0.5
            else:
                return self.transform(image)
        except TypeError:
            try:
                image = pImage.open(
                    BytesIO(requests.get(self.dataset[index][self.image_column], timeout=30).content)
                )
            except ConnectionError:
                try:
                    print("Image request failure, attempting next image")
                    index += 1

                    image = pImage.open(
                        BytesIO(requests.get(self.dataset[index][self.image_column], timeout=30).content)
                    )
                except ConnectionError:
                    raise ConnectionError("Unable to request image from the Dataset")

            if self.using_taming:
                return self.transform(image) - 0.5
            else:
                return self.transform(image)


class ImagecondDataset(ImageDataset):
    def __init__(
        self,
        dataset,
        image_size: int,
        image_column="image",
        cond_image_column="cond_image",
        flip=True,
        center_crop=True,
        stream=False,
        using_taming=False,
        random_crop=False,
        embeds=[],
    ):
        super().__init__(
            dataset,
            image_size=image_size,
            image_column=image_column,
            flip=flip,
            stream=stream,
            center_crop=center_crop,
            using_taming=using_taming,
            random_crop=random_crop,
        )
        self.cond_image_column = cond_image_column

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        cond_image = self.dataset[index][self.cond_image_column]
        
        return self.transform(image), self.transform(cond_image)


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def save_dataset_with_progress(dataset, save_path):
    # Estimate the total size of the dataset in bytes
    total_size = sys.getsizeof(dataset)

    # Start saving the dataset in a separate thread
    save_thread = Thread(target=dataset.save_to_disk, args=(save_path,))
    save_thread.start()

    # Create a tqdm progress bar and update it periodically
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        while save_thread.is_alive():
            if os.path.exists(save_path):
                size = get_directory_size(save_path)
                # Update the progress bar based on the current size of the saved file
                pbar.update(size - pbar.n)  # Update by the difference between current and previous size
            time.sleep(1)
def image_path_to_tensor(
        image_path
    ):
        image = pImage.open(image_path)
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

        return image_tensor

def find_image(data_root, target_name, extensions):
    for ext in extensions:
        for path in Path(data_root).rglob(f"*.{ext}"):  # 遞迴搜尋所有副檔名符合的檔案
            if target_name.lower() in path.stem.lower():  # 檢查名稱
                print(path)
                return path  # 找到就回傳
    return None

def get_dataset_from_dataroot(
    data_root,
    cond_image_root,
    image_column="image",
    cond_image_column="cond_image",
    save_path="dataset",
    save=True, 
):
    # Check if data_root is a symlink and resolve it to its target location if it is
    if os.path.islink(data_root):
        data_root = os.path.realpath(data_root)

    if os.path.exists(save_path):
        # Get the modified time of save_path
        save_path_mtime = os.stat(save_path).st_mtime

        if save:
            # Traverse the directory tree of data_root and get the modified time of all files and subdirectories
            print("Checking modified date of all the files and subdirectories in the dataset folder.")
            data_root_mtime = max(
                os.stat(os.path.join(root, f)).st_mtime
                for root, dirs, files in os.walk(data_root)
                for f in files + dirs
            )

            # Check if data_root is newer than save_path
            if data_root_mtime > save_path_mtime:
                print(
                    "The data_root folder has being updated recently. Removing previously saved dataset and updating it."
                )
                shutil.rmtree(save_path, ignore_errors=True)
            else:
                print("The dataset is up-to-date. Loading...")
                # Load the dataset from save_path if it is up-to-date
                return load_from_disk(save_path)

    extensions = ["jpg", "jpeg", "png", "webp"]
    image_paths = []
    cond_image_target_name = "attention"

    for ext in extensions:
        image_paths.extend(list(Path(data_root).rglob(f"*.{ext}")))

    random.shuffle(image_paths)
    data_dict = {image_column: [], cond_image_column: []}
    for image_path in tqdm(image_paths):
        # check image size and ignore images with 0 byte.
        if os.path.getsize(image_path) == 0:
            continue
        if "attention" in image_path.stem.lower():
            continue 
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]
        
        cond_image_path = find_image(image_path.parent, cond_image_target_name, extensions)
        #cond_image_path = os.path.join(cond_image_root, f"{image_name}.png")
        
        if os.path.exists(str(cond_image_path)):
            cond_image_tensor = image_path_to_tensor(cond_image_path)

        image_path = str(image_path)
        data_dict[image_column].append(image_path)
        data_dict[cond_image_column].append(str(cond_image_path))
    dataset = datasets.Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(image_column, Image())
    dataset = dataset.cast_column(cond_image_column, Image())

    if save:
        save_dataset_with_progress(dataset, save_path)

    return dataset

def split_dataset_into_dataloaders(dataset, valid_frac=0.05, seed=42, batch_size=1):
    print(f"Dataset length: {len(dataset)} samples")
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        print(f"Splitting dataset into {train_size} training samples and {valid_size} validation samples")
        split_generator = torch.Generator().manual_seed(seed)
        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, valid_size],
            generator=split_generator,
        )
    else:
        print("Using shared dataset for training and validation")
        train_dataset = dataset
        validation_dataset = dataset

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader, validation_dataloader





