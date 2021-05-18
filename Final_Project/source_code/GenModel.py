'''
Final Project
Team Members: Shirish Mecheri Vogga, Jasmeet Narang
'''


import os
import tensorflow
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Reshape, Concatenate, Activation
import tensorflow.keras.backend as K
from AnchorBoxes import AnchorBoxes
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from SSDLoss import SSDLoss
from get_data import get_data
from scipy.io import loadmat
import pandas as pd
from ssd_input_encoder import SSDInputEncoder
import ssd_output_decoder as decoder_op
from DataGenerator import DataGenerator
import math
from PIL import Image
import cv2 
# tensorflow.compat.v1.disable_eager_execution()
# def get_base_network():
#     base_network = VGG16(
#         weights='imagenet', include_top=False, input_shape=(640,640,3)
#     )
#
#     c3_output, c4_output, c5_output = [
#         base_network.get_layer(layer_name).output
#         for layer_name in ["block3_conv3", "block4_conv3", "block5_conv3"]
#     ]
#
#     return Model(
#         inputs = [base_network.inputs], outputs=[c3_output, c4_output, c5_output]
#     )
#
# inputs, outputs = get_base_network()
# print(inputs.shape)

# def get_anchors(height, weight):

img_height = 240# Height of the input images
img_width = 240 # Width of the input images
img_channels = 3 # Number of color channels of the input images
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1 # Number of positive classes
scales = [0.01, 0.16, 0.32, 0.64, 0.28, 2.56, 5.12] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [1.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size

def get_model(image_size = (240,240,3),
                n_classes=2,
                mode='training',
                l2_regularization=0.0001,
                min_scale=0.1,
                max_scale=0.9,
                scales=[1,16, 32, 64, 128, 256, 512],
                aspect_ratios_global=[1.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='minmax',
                normalize_coords=True,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False):
    n_predictor_layers = 6
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:  # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")


    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    base_network = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(240,240, 3))

    base_network.trainable = False

    c3_output, c4_output, c5_output = [
        base_network.get_layer(layer_name).output
        for layer_name in ["block3_conv3", "block4_conv3", "block5_conv3"]]

    vgg_16_output = base_network.output

    fc6 = layers.Conv2D(1024, 3, padding='same',activation='relu')(vgg_16_output)

    fc7 = layers.Conv2D(1024, 1, padding='same', activation='relu')(fc6) #detection layer

    conv6_1 = layers.Conv2D(256, 1, activation='relu')(fc7)

    conv6_2 = layers.Conv2D(512, 3, strides=2, activation='relu')(conv6_1) #detection layer

    conv7_1 = layers.Conv2D(128, 1, activation='relu')(conv6_2)

    conv7_2 = layers.Conv2D(256, 3, strides=2, activation='relu')(conv7_1) #detection layer

    normalized_c3 = Lambda(lambda t: K.l2_normalize(10 * t, axis=[1, 2, 3]))(c3_output)

    normalized_c4 = Lambda(lambda t: K.l2_normalize(8 * t, axis=[1, 2, 3]))(c4_output)

    normalized_c5 = Lambda(lambda t: K.l2_normalize(5 * t, axis=[1, 2, 3]))(c5_output)

    # p1 = layers.Conv2D(6, 3, activation='relu')(normalized_c3)  # units = Ns+4

    # p2 = layers.Conv2D(6, 3, activation='relu')(normalized_c4)

    # p3 = layers.Conv2D(6, 3, activation='relu')(normalized_c5)

    # p4 = layers.Conv2D(6, 3, activation='relu')(fc7)

    # p5 = layers.Conv2D(6, 3, activation='relu')(conv6_2)

    # p6 = layers.Conv2D(6, 3, activation='relu')(conv7_2)

    classes1 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes1')(normalized_c3)
    classes2 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg), name='classes2')(normalized_c4)
    classes3 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg), name='classes3')(normalized_c5)
    classes4 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg), name='classes4')(fc7)
    classes5 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg), name='classes5')(conv6_2)
    classes6 = layers.Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2_reg), name='classes6')(conv7_2)

    boxes1 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes1')(normalized_c3)
    boxes2 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes2')(normalized_c4)
    boxes3 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes3')(normalized_c5)
    boxes4 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes4')(fc7)
    boxes5 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes5')(conv6_2)
    boxes6 = layers.Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes6')(conv7_2)


    anchors1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios_global,
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors1')(boxes1)
    anchors2 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios_global,two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors2')(boxes2)
    anchors3 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],aspect_ratios=aspect_ratios_global, two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors3')(boxes3)
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios_global, two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios_global,two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                           aspect_ratios=aspect_ratios_global, two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords,
                           name='anchors6')(boxes6)


    classes1_reshaped = Reshape((-1, n_classes), name='classes1_reshape')(classes1)
    classes2_reshaped = Reshape((-1, n_classes), name='classes2_reshape')(classes2)
    classes3_reshaped = Reshape((-1, n_classes), name='classes3_reshape')(classes3)
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)


    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes1_reshaped = Reshape((-1, 4), name='boxes1_reshape')(boxes1)
    boxes2_reshaped = Reshape((-1, 4), name='boxes2_reshape')(boxes2)
    boxes3_reshaped = Reshape((-1, 4), name='boxes3_reshape')(boxes3)
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors1_reshaped = Reshape((-1, 8), name='anchors1_reshape')(anchors1)
    anchors2_reshaped = Reshape((-1, 8), name='anchors2_reshape')(anchors2)
    anchors3_reshaped = Reshape((-1, 8), name='anchors3_reshape')(anchors3)
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)


    classes_concat = Concatenate(axis=1, name='classes_concat')([classes1_reshaped,
                                                                 classes2_reshaped,
                                                                 classes3_reshaped,
                                                                 classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes1_reshaped,
                                                             boxes2_reshaped,
                                                             boxes3_reshaped,
                                                             boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors1_reshaped,
                                                                 anchors2_reshaped,
                                                                 anchors3_reshaped,
                                                                 anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped])


    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    model = Model(inputs=base_network.inputs, outputs=predictions)
    return model

print("before model")
model = get_model()
print("getting summary")
print(model.summary())

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3,alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

model.load_weights("Models/model_weights1_epoch_155.h5")

img_path_train = "D:/Personal/Computer Vision/Final_Project/WIDER_train/images"
img_path_val = "D:/Personal/Computer Vision/Final_Project/WIDER_val/images"
#img_path_test = "C:\\Users\jasme\Desktop\WIDER_test\WIDER_test\images"

list_files_train1 = os.listdir(img_path_train)

list_files_val1 = os.listdir(img_path_val)

#list_files_test1 = os.listdir(img_path_test)


print("list_files_train: ", list_files_train1)

# train_dataset = np.array(zip(get_data.train_bounding_box, get_data.test_invalid_labels, get_data.train_invalid_labels, get_data.train_files_list))
# train dataset features and labels
train_dataset = loadmat("D:/Personal/Computer Vision/Final_Project/wider_face_split/wider_face_train.mat")
train_bounding_box, train_invalid_labels, train_files_list, ignore = get_data(train_dataset,list_files_train1)

train1 = pd.DataFrame(data=train_bounding_box)
train2 = pd.DataFrame(data=train_invalid_labels)
train3 = pd.DataFrame(data=train_files_list)
train_set = pd.concat([train3, train1, train2], axis=1, sort=False)

train_set = pd.DataFrame(train_set)

train_set.to_csv('train_data.csv',index=False)

train_set = "train_data.csv"

# train_set = train_set.to_numpy()
# print("##########")
# print(train_set)

#validation dataset features and labels
val_dataset = loadmat("D:/Personal/Computer Vision/Final_Project/wider_face_split/wider_face_val.mat")
val_bounding_box, val_invalid_labels, val_files_list, ignore = get_data(val_dataset,list_files_val1)

val1 = pd.DataFrame(data=val_bounding_box)
val2 = pd.DataFrame(data=val_invalid_labels)
val3 = pd.DataFrame(data=val_files_list)
val_set = pd.concat([val3, val1, val2], axis=1, sort=False)

val_set = pd.DataFrame(val_set)

val_set.to_csv('val_data.csv',index=False)

val_set ="val_data.csv"
train_set = "train_data.csv"
# val_set = val_set.to_numpy()


#test dataset features and labels
#test_dataset = loadmat("C:\\Users\jasme\Desktop\wider_face_split\wider_face_split\wider_face_test.mat")
#test_bounding_box, test_invalid_labels, ignore, test_files_list = get_data(test_dataset, list_files_test1)

train_generator = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
print("generator: ", train_generator)
val_generator = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

train_generator.parse_csv(images_dir=img_path_train,
                        labels_filename=train_set,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

train_dataset_size = train_generator.get_dataset_size()



val_generator.parse_csv(images_dir=img_path_val,
                        labels_filename=val_set,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])


val_dataset_size = val_generator.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

predictor_sizes = [model.get_layer('classes1').output_shape[1:3],
                   model.get_layer('classes2').output_shape[1:3],
                   model.get_layer('classes3').output_shape[1:3],
                   model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.35,
                                    neg_iou_limit=0.2,
                                    coords='minmax',
                                    normalize_coords=normalize_coords)

batch_size = 32
train_generator1 = train_generator.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator1 = val_generator.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

count = 0
temp = next(train_generator1)
boxes = temp[1]
for i in range(boxes.shape[0]):
    for j in range(boxes.shape[1]):
        for k in range(boxes.shape[2]):
            if boxes[i][j][1] == 1:
                count+=1
                #print("positive class detected")
initial_epoch   = 0
final_epoch     = 10
steps_per_epoch = math.ceil(train_dataset_size/batch_size)
    # 100
    # math.ceil(train_dataset_size/batch_size)


# for i in range(len(x_train)):
#     temp = next(train_generator1)
#     temp1 = next(val_generator1)
#
#     x_train = temp[0]
#     y_train = temp[1]
#
#     # print("pred layer: ", predictor_sizes)
#
#     x_val = temp1[0]
#     y_val = temp1[1]
print("val steps: ", math.ceil(val_dataset_size/batch_size))
# history = model.fit(x_train, y_train,
#                 steps_per_epoch=steps_per_epoch,
#                 epochs=final_epoch,
#                 callbacks=None,
#                 validation_data=(x_val, y_val),
#                 validation_steps=math.ceil(val_dataset_size/batch_size),
#                 initial_epoch=initial_epoch)
for i in range(2):
    history = model.fit_generator(train_generator1,
                    steps_per_epoch=steps_per_epoch,
                    epochs=final_epoch,
                    callbacks=None,
                    validation_data=val_generator1,
                    validation_steps=math.ceil(val_dataset_size/batch_size),
                    initial_epoch=initial_epoch)
    model.save_weights("Models/model_weights1_epoch_50.h5")

    
    
print("loss: ", history.history['loss'])
print("val loss: ", history.history['val_loss'])

model.save("/content/gdrive/MyDrive/project/model1_epoch_5.h5")

model.load_weights("model_weights1_epoch_5.h5")
#Plot the predictions 
predict_generator = val_generator.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)


batch_images, batch_labels, batch_filenames = next(predict_generator)


i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])

batch_labels = np.array(batch_labels)


y_pred = model.predict(batch_images)

y_pred_decoded = decoder_op.decode_detections_fast(y_pred,
                                   confidence_thresh=0.05,
                                   iou_threshold=0.35,
                                   top_k=400,
                                   normalize_coords=True,
                                   input_coords='minmax',
                                   img_height=img_height,
                                   img_width=img_width)

#y_pred_decoded = decoder_op.decode_detections(y_pred,
#                                   confidence_thresh=0.5,
 #                                  iou_threshold=0.45,
  #                                 top_k=400,
    #                               normalize_coords=True,
   #                                input_coords='centroids',
     #                              img_height=img_height,
      #                             img_width=img_width)


import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
img = cv2.imread(batch_filenames[0])
#image = batch_image[i].resize((1000,1000))
plt.imshow(img)

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
classes = ['Face'] # Just so we can print class names onto the image instead of IDs

# Draw the ground truth boxes in green (omit the label for more clarity)
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin,"face", size='x-large', color='red', bbox={'facecolor':'green', 'alpha':1.0})

# Draw the predicted boxes in blue
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    #color = colors[int(box[0])]
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='yellow', fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, "face", size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})


