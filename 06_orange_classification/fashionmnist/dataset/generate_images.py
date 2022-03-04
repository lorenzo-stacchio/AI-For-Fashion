import gzip
import os.path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def check_dir_ex(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


main_folder = "C:/Users/Chiqu/Documents/Progetti/AI-For-Fashion/06_orange_classification/fashionmnist/dataset/"
image_folder = "D:/Download/FashionMnist_images/"
image_size = 28

dict_num_classes = {0: "T-shirt_top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}


def read_data(folder, file_images, file_labels, num_images, out_folder):
    images = gzip.open(folder + file_images, 'r')
    images.read(16)  # Read file header
    buf = images.read(image_size * image_size * num_images)  # Read images bytes
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = images.reshape(num_images, image_size, image_size, 1)
    labels = gzip.open(folder + file_labels, 'r')
    labels.read(8)  # Read file header
    buf = labels.read(image_size * image_size * num_images)  # Read images bytes
    labels = np.frombuffer(buf, dtype=np.uint8)
    # Create folder
    for l in set(labels):
        check_dir_ex(out_folder + "%s/" % dict_num_classes[l])

    for idx, (image, label) in enumerate(zip(images, labels)):
        image = np.asarray(image).squeeze()
        cv2.imwrite(out_folder + "%s/%s.jpg" % (dict_num_classes[label], idx), image)
        # plt.title(dict_num_classes[label])
        # plt.imshow(image)
        # plt.show()


read_data(main_folder, 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 60000, image_folder + "train/")
read_data(main_folder, 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 10000, image_folder + "test/")
