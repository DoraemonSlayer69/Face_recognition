

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
#import cv2
import csv
import os
import sys
from tqdm import tqdm, trange

try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from ssd_input_encoder import SSDInputEncoder
from AnchorBoxes import iou

class BoundGenerator:
    '''
    Generates pairs of floating point values that represent lower and upper bounds
    from a given sample space.
    '''
    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):
        '''
        Arguments:
            sample_space (list or tuple): A list, tuple, or array-like object of shape
                `(n, 2)` that contains `n` samples to choose from, where each sample
                is a 2-tuple of scalars and/or `None` values.
            weights (list or tuple, optional): A list or tuple representing the distribution
                over the sample space. If `None`, a uniform distribution will be assumed.
        '''

        if (not (weights is None)) and len(weights) != len(sample_space):
            raise ValueError("`weights` must either be `None` for uniform distribution or have the same length as `sample_space`.")

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError("All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None: bound_pair[0] = 0.0
            if bound_pair[1] is None: bound_pair[1] = 1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError("For all sample space elements, the lower bound cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0/self.sample_space_size] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        '''
        Returns:
            An item of the sample space, i.e. a 2-tuple of scalars.
        '''
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]

class BoxFilter:
    '''
    Returns all bounding boxes that are valid with respect to a the defined criteria.
    '''

    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Arguments:
            check_overlap (bool, optional): Whether or not to enforce the overlap requirements defined by
                `overlap_criterion` and `overlap_bounds`. Sometimes you might want to use the box filter only
                to enforce a certain minimum area for all boxes (see next argument), in such cases you can
                turn the overlap requirements off.
            check_min_area (bool, optional): Whether or not to enforce the minimum area requirement defined
                by `min_area`. If `True`, any boxes that have an area (in pixels) that is smaller than `min_area`
                will be removed from the labels of an image. Bounding boxes below a certain area aren't useful
                training examples. An object that takes up only, say, 5 pixels in an image is probably not
                recognizable anymore, neither for a human, nor for an object detection model. It makes sense
                to remove such boxes.
            check_degenerate (bool, optional): Whether or not to check for and remove degenerate bounding boxes.
                Degenerate bounding boxes are boxes that have `xmax <= xmin` and/or `ymax <= ymin`. In particular,
                boxes with a width and/or height of zero are degenerate. It is obviously important to filter out
                such boxes, so you should only set this option to `False` if you are certain that degenerate
                boxes are not possible in your data and processing chain.
            overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within the given `overlap_bounds`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within the given `overlap_bounds`.
            overlap_bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            min_area (int, optional): Only relevant if `check_min_area` is `True`. Defines the minimum area in
                pixels that a bounding box must have in order to be valid. Boxes with an area smaller than this
                will be removed.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        '''
        if not isinstance(overlap_bounds, (list, tuple, BoundGenerator)):
            raise ValueError("`overlap_bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(overlap_bounds, (list, tuple)) and (overlap_bounds[0] > overlap_bounds[1]):
            raise ValueError("The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError("`overlap_criterion` must be one of 'iou', 'area', or 'center_point'.")
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        '''
        Arguments:
            labels (array): The labels to be filtered. This is an array with shape `(m,n)`, where
                `m` is the number of bounding boxes and `n` is the number of elements that defines
                each bounding box (box coordinates, class ID, etc.). The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): Only relevant if `check_overlap == True`. The height of the image
                (in pixels) to compare the box coordinates to.
            image_width (int): `check_overlap == True`. The width of the image (in pixels) to compare
                the box coordinates to.

        Returns:
            An array containing the labels of all boxes that are valid.
        '''

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Record the boxes that pass all checks here.
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:

            non_degenerate = (labels[:,xmax] > labels[:,xmin]) * (labels[:,ymax] > labels[:,ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:

            min_area_met = (labels[:,xmax] - labels[:,xmin]) * (labels[:,ymax] - labels[:,ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:

            # Get the lower and upper bounds.
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # Compute which boxes are valid.

            if self.overlap_criterion == 'iou':
                # Compute the patch coordinates.
                image_coords = np.array([0, 0, image_width, image_height])
                # Compute the IoU between the patch and all of the ground truth boxes.
                image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]], coords='corners', mode='element-wise', border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou > lower) * (image_boxes_iou <= upper)

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
                elif self.border_pixels == 'exclude':
                    d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
                # Compute the areas of the boxes.
                box_areas = (labels[:,xmax] - labels[:,xmin] + d) * (labels[:,ymax] - labels[:,ymin] + d)
                # Compute the intersection area between the patch and all of the ground truth boxes.
                clipped_boxes = np.copy(labels)
                clipped_boxes[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=image_height-1)
                clipped_boxes[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=image_width-1)
                intersection_areas = (clipped_boxes[:,xmax] - clipped_boxes[:,xmin] + d) * (clipped_boxes[:,ymax] - clipped_boxes[:,ymin] + d) # +1 because the border pixels belong to the box areas.
                # Check which boxes meet the overlap requirements.
                if lower == 0.0:
                    mask_lower = intersection_areas > lower * box_areas # If `self.lower == 0`, we want to make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
                else:
                    mask_lower = intersection_areas >= lower * box_areas # Especially for the case `self.lower == 1` we want the ">=" sign, otherwise no boxes would count at all.
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # Compute the center points of the boxes.
                cy = (labels[:,ymin] + labels[:,ymax]) / 2
                cx = (labels[:,xmin] + labels[:,xmax]) / 2
                # Check which of the boxes have center points within the cropped patch remove those that don't.
                requirements_met *= (cy >= 0.0) * (cy <= image_height-1) * (cx >= 0.0) * (cx <= image_width-1)

        return labels[requirements_met]


class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass

class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data: A general-purpose CSV parser,
    an XML parser for the Pascal VOC datasets, and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        '''
        Initializes the data generator. You can either load a dataset directly here in the constructor,
        e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
                format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
                don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
                data.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

        self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None # The only way that this list will not stay `None` is if `load_images_into_memory == True`.

        # `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves. This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else: it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None


    

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located.
            input_format (list): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
                are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, and image IDs.
        '''

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError("`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread) # Skip the header row.
            for row in csvread: # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes: # If the class_id is among the classes that are to be included in the dataset...
                    box = [] # Store the box class and coordinates here
                    box.append(row[self.input_format.index('image_name')].strip()) # Select the image name column in the input format and append its content to `box`
                    for element in self.labels_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[0] # The image ID will be the portion of the image name before the first dot.
        current_labels = [] # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):

            if box[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else: # If this box belongs to a new image file
                if random_sample: # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                current_labels = [] # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret: # In case we want to return these
            return self.images, self.filenames, self.labels, self.image_ids

    

    

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
                it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                    but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                    in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                    bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                    or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                    usually treated to be neutral in a model evaluation.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        '''

        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

       

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

       

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

      

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0

           

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = self.dataset_indices[current:current+batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            elif not (self.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current+batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        image = image.resize((240, 240))
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current+batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################
            # print("in generator: ", batch_y_encoded)
            # print("in generator type: ", type(batch_y_encoded))
            # print("in generator shape: ", batch_y_encoded.shape)
            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            # ret = np.asarray(ret).astype('float32')
            ret = tuple(ret)
            yield ret

   

    def get_dataset(self):
        '''
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        '''
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return self.dataset_size
