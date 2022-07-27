import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import utils.binvox_rw as binvox_rw
import os
import pandas as pd

label_file = pd.read_excel('Multifeature labels.xls')
print(label_file.head())

id_lable = {0: "None", 1: "Through hole", 2: "Blind hole", 3: "Triangular Passage", 4: "Rectangular Passage",
                5: "Circular through slot", 6: "Triangular through slots", 7: "Circular through slot",
                8: "rectangular through step", 9: "Triangular Pocket", 10: "Rectangular pocket",
                11: "Circular end pocket", 12: "Triangular Blind Step", 13: "Circular blind step",
                14: "Rectangular blind step", 15: "Rectangular through step", 16: "2-Sided through step",
                17: "Slanted through step", 18: "Chamfer", 19: "Round",
                20: "Vertical circular blind slot", 21: "Horizontal Circular End Blind Slot",
                22: "6-Sided Passage", 23: "6-Side pocket", 24: "Round (Chamfer class)", 25:"Rectangular blind slot",
                26: "ORing", 27: "Rectangular through slot", 28: "Circular end step", 29:"Passage", 30: "(GEAR)",
                31: "7-Side pocket"}
lable_id = {"None":0, "Through hole": 1 , "Blind hole": 2, "Triangular Passage": 3, "Rectangular Passage": 4,
                "Circular through slot": 5, "Triangular through slots": 6, "Circular through slot": 7,
                "rectangular through step": 8, "Triangular Pocket": 9, "Rectangular pocket": 10,
                "Circular end pocket": 11, "Triangular Blind Step": 12, "Circular blind step" : 13,
                "Rectangular blind step": 14, "Rectangular through step": 15, "2-Sided through step": 16,
                "Slanted through step": 17, "Chamfer": 18, "Round": 19,
                "Vertical circular blind slot": 20, "Horizontal Circular End Blind Slot": 21,
                "6-Sided Passage": 22, "6-Side pocket": 23, "Round (Chamfer class)": 24, "Rectangular blind slot": 25,
                "ORing": 26, "Rectangular through slot": 27, "Circular end step": 28, "Passage": 29, "(GEAR)": 30,
                "7-Side pocket": 31}

def write_batches_for_split(split_name, sample_keys, resolution, norm):
    batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
    labels = []
    names = []
    batch_idx = 0
    num_of_batches = 0

    file_name = split_name + '.h5'
    hf = h5py.File(file_name, 'a')

    for key in sample_keys:
        x, y = read_voxel_from_binvox(key, norm)

        batch[batch_idx, :, :, :, :] = x
        batch_idx += 1
        labels.append(y)
        names.append(key)

        if batch_idx == batch_size:
            print(f"Batch Num: {num_of_batches}")
            group = hf.create_group(str(num_of_batches))
            cad_names = np.array(names, dtype="S")
            group.create_dataset("names", data=cad_names, compression="lzf")
            group.create_dataset("x", data=batch, compression="lzf")
            group.create_dataset("y", data=labels)

            batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
            batch_idx = 0
            num_of_batches += 1
            labels = []
            names = []

    hf.close()


def read_h5(split):
    hf = h5py.File(split + ".h5", 'r')
    labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
              16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

    for key in list(hf.keys()):
        group = hf.get(key)
        x = np.array(group.get("x"), dtype=np.float32)
        y = np.array(group.get("y"), dtype=np.int8)

        print(f"Group: {group}")
        print(f"X: {np.shape(x)}")
        print(f"Y: {y}")

        for label in y:
            labels[label] += 1

    hf.close()

    print(labels)


def split_dataset(split, samples):
    random.shuffle(samples)
    random.shuffle(samples)

    train_idx = int(math.ceil(split["train"] * len(samples)))
    val_idx = int(math.ceil((split["val"] * len(samples))) + train_idx)

    train_list = samples[:train_idx]
    val_list = samples[train_idx:val_idx]
    test_list = samples[val_idx:]

    return train_list, val_list, test_list


def get_label(filename):
    temp_df = label_file.loc[label_file['PART NAME'] == filename]
    id1 = lable_id[temp_df['LABELS1'].iloc[0]] if not pd.isna(temp_df['LABELS1'].iloc[0]) else 0
    id2 = lable_id[temp_df['LABELS2'].iloc[0]] if not pd.isna(temp_df['LABELS2'].iloc[0]) else 0
    id3 = lable_id[temp_df['LABELS3'].iloc[0]] if not pd.isna(temp_df['LABELS3'].iloc[0]) else 0
    print(temp_df['LABELS3'])
    id4 = lable_id[temp_df['LABELS4'].iloc[0]] if not pd.isna(temp_df['LABELS4'].iloc[0]) else 0
    id5 = lable_id[temp_df['LABELS5'].iloc[0]] if not pd.isna(temp_df['LABELS5'].iloc[0]) else 0
    return (id1, id2, id3, id4, id5)

def read_voxel_from_binvox(filepath, normalize=True):
    with open(filepath, "rb") as f:
        model = binvox_rw.read_as_3d_array(f)
    voxel = model.data

    if normalize:
        voxel = zero_centering_norm(voxel)

    filename = filepath.split("\\")[-1]
    label = get_label(filename.split(".")[0])
    # label = int(filename.split("_")[0])
   # label = int(filename.split("-")[0])

    # array size 5
    # (1,3,6,8,9)
    # (1, 3, 6, 8, 9)

    voxel = np.array(voxel, dtype=np.float32)
    label = np.array(label, dtype=np.int8)

    return voxel, label


def zero_centering_norm(voxels):
    norm = (voxels - 0.5) * 2
    return norm


def display_voxel(voxels):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, edgecolor='k')
    plt.show()


if __name__ == '__main__':
    # Parameters to set

    main_dir = '{}\\utils\\VM_TRAIN_DATA\\'.format(os.getcwd())
    batch_size = 40
    voxel_resolution = 64
    normalize = True
    dataset_split = {"train": 0.7, "val": 0.15, "test": 0.15}

    """
    list_of_files = []
    for i in os.listdir(main_dir):
        sub_dir = main_dir + i + "/"
        list_of_files.extend(glob.glob(sub_dir + "*.binvox"))
    """
    list_of_files = glob.glob(main_dir + "*.binvox")

    print(list_of_files)



    train_samples, val_samples, test_samples = split_dataset(dataset_split, list_of_files)

    print("Train")
    write_batches_for_split("train", train_samples, voxel_resolution, normalize)
    print("Validation")
    write_batches_for_split("val", val_samples, voxel_resolution, normalize)
    print("Test")
    write_batches_for_split("test", test_samples, voxel_resolution, normalize)
