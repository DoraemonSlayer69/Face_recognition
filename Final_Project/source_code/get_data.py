import numpy as np
from scipy.io import loadmat
import os
import bounding_box_utils as box_stuff

def get_data(dataset, list_files):
    bounding_box = []
    invalid_label = []

    #as test dataset does not contain face_bbx_list and labels
    try:
        bounding_box = list(dataset['face_bbx_list'])
        invalid_label = list(dataset['invalid_label_list'])
    except KeyError:
        pass

    file_list = dataset['file_list']
    # print("file: ", file_list)

    files_list = []
    fs="/"
    for r in range(len(file_list)):
        for i in range(len(file_list[r][0])):
            temp = file_list[r][0][i][0]
            filename = temp[0]
            files_list.append(list_files[r]+fs+filename+".jpg")
            # for f in range(len(files)):
            #     files_list.append(files[f])
    # files_list = np.array(files_list)

    new_file_list = []
    no_bounding_boxes = []
    # get the  bounding box
    bounding_boxes = []
    for r1 in range(len(bounding_box)):
        for i1 in range(len(bounding_box[r1][0])):
            boxes = bounding_box[r1][0][i1][0]
            no_bounding_boxes.append(len(bounding_box[r1][0][i1][0]))

            for b in range(len(boxes)):
                bounding_boxes.append(boxes[b])
                # print("r1: ", r1)
                # print(len(bounding_box[r1][0][i1][0]))
                # new_file_list.append(files_list[r1][i1])
                # print(new_file_list)

    # print("files_list: ",len(files_list))
    # print("bounding box: ",len(no_bounding_boxes))
    for i, k in zip(files_list, no_bounding_boxes):

        for s in range(k):
            # temp = i
            # print("i : ", i)
            # print("s: ", s)
            new_file_list.append(i)

    new_file_list = np.array(new_file_list)

    # print(type(bounding_boxes))
    bounding_boxes = np.array(bounding_boxes)
    print(bounding_boxes.shape)
    bounding_box_new = box_stuff.convert_coordinates(bounding_boxes,0,"centroids2minmax")
    bounding_box_new = bounding_box_new.astype('int32')
    # print("bounding shape: ", bounding_boxes.shape)

    # invalid label extraction
    invalid_labels = []
    for r2 in range(len(invalid_label)):
        for i2 in range(len(invalid_label[r2][0])):
            labels = invalid_label[r2][0][i2][0]
            for l in range(len(labels)):
                if labels[l] == 0:
                    invalid_labels.append(1)
                else:
                    invalid_labels.append(0)


    # print(type(invalid_labels))
    invalid_labels = np.array(invalid_labels)
    # print(invalid_labels)
    # print("invalid shape: ", invalid_labels.shape)


    # extracting all file paths
    # print(file_list[0][0][0][0])

    # files_list = []
    # for r in range(len(file_list)):
    #     for i in range(len(file_list[r][0])):
    #         files = file_list[r][0][i]
    #         for f in range(len(files)):
    #             files_list.append(files[f])
    # files_list = np.array(files_list)
    # print(files_list)
    # print(files_list.shape)

    return bounding_box_new, invalid_labels, new_file_list, np.array(files_list)

"""
img_path_train = "C:\\Users\jasme\Desktop\WIDER_train\WIDER_train\images"
img_path_val = "C:\\Users\jasme\Desktop\WIDER_val\WIDER_val\images"
list_files_train = os.listdir(img_path_train)
list_files_val = os.listdir(img_path_val)

#train dataset features and labels
train_dataset = loadmat("C:\\Users\jasme\Desktop\wider_face_split\wider_face_split\wider_face_train.mat")
train_bounding_box, train_invalid_labels, train_files_list, ignore = get_data(train_dataset, )


#validation dataset features and labels
val_dataset = loadmat("C:\\Users\jasme\Desktop\wider_face_split\wider_face_split\wider_face_val.mat")
val_bounding_box, val_invalid_labels, val_files_list, ignore = get_data(val_dataset)

#test dataset features and labels
test_dataset = loadmat("C:\\Users\jasme\Desktop\wider_face_split\wider_face_split\wider_face_test.mat")
test_bounding_box, test_invalid_labels, ignore, test_files_list = get_data(test_dataset)

print(test_files_list.shape)
print(test_invalid_labels.shape)

"""
