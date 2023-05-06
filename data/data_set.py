from cv2 import imread, resize, imshow, destroyAllWindows, waitKey
import numpy as np
import torch
import glob, os
import json
import xml.etree.cElementTree as ET
from parameters import IMAGE_HEIGHT, IMAGE_WIDTH, device


class CustomImageDataset:
    def __init__(
        self, base_path="Single_Labels", img_dir=glob.glob("Line_data/Images/*/*/*"),
    ):

        self.base_path = base_path
        self.img_dir = img_dir

    def Load_Image_Label(self, image_path):
        # Open the image file
        label = tuple()
        json_path = os.path.join(
            self.base_path, image_path.split("\\")[-1][:-4] + ".json"
        )
        with open(json_path, "r") as json_file:
            label = json.load(json_file)
        img = imread(image_path, 0)
        img = 255 - img
        img_height, img_width = img.shape[0], img.shape[1]
        n_repeats = int(np.ceil(IMAGE_WIDTH / img_width))
        padded_image = np.concatenate([img] * n_repeats, axis=1)
        padded_image = padded_image[:IMAGE_HEIGHT, :IMAGE_WIDTH]
        resized_img = resize(padded_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        return (resized_img, label)
        # plt.imshow(img)
        # plt.show()

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        print(idx)
        # import pdb;pdb.set_trace()
        Image, Labels = self.Load_Image_Label(self.img_dir[idx])
        return torch.tensor(Image, device="cpu").float(), Labels
        # return Image,Labels
