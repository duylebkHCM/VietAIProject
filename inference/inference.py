import time
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.compat.v1 as tfc
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

class Config():
    MODEL_NAME = 'custom_ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
    PATH_TO_MODEL_DIR = os.path.join('workspace/training/exported-models', MODEL_NAME)
    PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
    PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
    PATH_TO_LABELS = 'workspace/training/annotations/label_map.pbtxt'
    PATH_TO_IMAGES = 'workspace/training/images/test'
    IMAGES_OUTPUT = 'workspace/inference/output'

config = Config()

print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection modeljoin()
configs = config_util.get_configs_from_pipeline_file(config.PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(config.PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def save_image_array_as_jpg(image, output_path):
    """Saves an image (represented as a numpy array) to JPG.

    Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    image_pil.save(output_path)

def main():

    category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)

    for image_path in glob.glob(config.PATH_TO_IMAGES + '/*.jpg'):

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

        # plt.figure()
        # plt.imshow(image_np_with_detections)
        image_name = image_path.split('/')[-1]
        save_image_array_as_jpg(image_np_with_detections, os.path.join(config.IMAGES_OUTPUT, image_name))
        print('Done')

    # plt.show()

    # sphinx_gallery_thumbnail_number = 2

if __name__ == '__main__':
    main()