import os.path
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
random.seed(42)

def check_dir_ex(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


image_folder = "D:/Download/FashionMnist_images/test/"
image_folder_out = "D:/Download/FashionMnist_images/test_reduced/"
image_size = 28

dict_num_classes = {0: "T-shirt_top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}


sample = 100

for class_dir in glob.glob(image_folder + "*"):
    images_path = [x for x in glob.glob(class_dir + "/*")]
    path_sample = random.sample(images_path, sample)
    class_out_dir = image_folder_out + os.path.basename(class_dir) + "/"
    check_dir_ex(class_out_dir)
    for path in path_sample:
        shutil.copy(path, class_out_dir + os.path.basename(path))
    print(class_dir, len(images_path))

