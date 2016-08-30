'''

author: @mlakhal

'''
import os
import cv2
import numpy as np
import h5py

from fnmatch import fnmatch
from progress.bar import Bar

WIDTH = 32
HEIGHT = 32
BATCH_SIZE = 1000

def count_img(path, ext):
    """ Return the number of 'ext' files
    fot a given folder.

    Args:
        path (str)  : path of the folder
        ext (str)   : extension of the file

    Returns:
        int     : number of files.

    """
    nf = 0
    for file in os.listdir(path):
        if fnmatch(file, ext):
            nf += 1

    return nf

def createDataset(path, batch_size, batch_name, rgb=True):
    """Create a new datset or build upon existing one.

    Args:
        path (str)      : path to the dataset folder
        len_batch(int)  : size of the batch
        batch_name(str) : name of the batch
        rgb (boolean)   : working with rgb images

    Returns:
        numpy array : new dataset

    """
    nf = count_img(path, "*.jpg")
    bar = Bar("Processing", max=nf)
    print("Building the dataset")

    cnt = 0; i = 0; ds = []
    for file in os.listdir(path):
        if fnmatch(file, "*.jpg"):
            img_path = path + "/" + file
            img = cv2.resize(cv2.imread(img_path), (WIDTH, HEIGHT)).astype(np.float32)
            bar.next()
            if not rgb and len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)
                i += 1
                ds.append(img)
            if rgb and len(img.shape) == 3:
                img = img.transpose((2,0,1))
                i += 1
                ds.append(img)
            if i % batch_size == 0:
                ds_name = batch_name + "_" + str(cnt) + ".h5"
                with h5py.File(ds_name, "w") as hf:
                    hf.create_dataset("dataset_1", data=np.array(ds))
                print("\nCreating batch N: {}".format(cnt))
                cnt += 1; ds = []
    if i % batch_size != 0:
        ds_name = batch_name + "_" + str(cnt) + ".h5"
        with h5py.File(ds_name, "w") as hf:
            hf.create_dataset("dataset_1", data=np.array(ds))
        print("\nCreating batch N: {}".format(cnt))
    bar.finish()


def main():
    if not os.path.exists("ds"):
        try:
            os.mkdir("ds")
        except OSError as error:
            print("Program error.\n{}".format(error))

    """
    path = "dataset/training/COMP"
    createDataset(path, BATCH_SIZE, "ds/ds_COMP")
    """
    path = "dataset/training/NOCOMP"
    createDataset(path, BATCH_SIZE, "ds/ds_NOCOMP")

if __name__ == "__main__":
    main()
