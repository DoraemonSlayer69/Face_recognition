import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    '''
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    '''

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # Compute the IoU.

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1), reps=(1,n))
        boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0), reps=(m,1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d)
        boxes2_areas = (boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas

class AnchorBoxes(Layer):
    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[1.0],
                 two_boxes_for_ar1=False,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):

        if K.backend() != 'tensorflow':
            raise TypeError(
                "This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(
                    K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError(
                "`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                    this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # Compute the number of boxes per cell
        if 1 in aspect_ratios and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean of this scale value and the next.
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height // feature_map_height
            step_width = self.img_width // feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                               border_pixels='half')
        elif self.coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax',
                                               border_pixels='half')

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(
            boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
