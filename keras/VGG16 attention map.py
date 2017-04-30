import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_cam

def main():
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    heatmaps = []
    for path in next(os.walk('./visual'))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img('./visual/' + path, target_size=(224, 224))
        pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, text=path)
        heatmaps.append(heatmap)

    cv2.imwrite("./grad-CAM visualization.jpg", utils.stitch_images(heatmaps))

if __name__ == "__main__":
    main()