import json
import sys
from glob import glob

from PIL import Image
from keras.models import load_model

from VGG16_ft import VGG16FineTuned
from bias_metric import compute_metric
from img_processing import dataset_convertor
from mnist_model import create_n_run_mnist
from model_utils import get_outputs_generator
from plant_village_custom_model import *
from numpy import argmax
from keras.applications.imagenet_utils import preprocess_input
import time
import pyximport; pyximport.install()
from heatmapgenerate import *


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf
# http://lcn.epfl.ch/tutorial/english/perceptron/html/learning.html
# https://github.com/fchollet/keras/issues/4446


def main():
    np.random.seed(123)  # for reproducibility
    random.seed(123)
    print("Keras bias v1.0")

    argv = sys.argv
    if argv[1] == "0":
        print("SEED IS", 123)
        vggft = VGG16FineTuned(dataset_loader=DatasetLoader(argv[2], int(argv[6])), mode=argv[4])
        vggft.train(int(argv[5]), weights_out=argv[3])
    # ==================================================================================================
    if argv[1] == "1":  # FIXME Cythonize - no need?
        dl = DatasetLoader(argv[2], 10000)
        model = load_model(argv[3])
        print("images to process:", dl.number_of_imgs_for_test)
        generate_maps(dl, model, argv[4], all_classes=bool(int(argv[5])), batch_size=int(argv[6]), mode=argv[7])

    if argv[1] == '2':
        dl = DatasetLoader(argv[3], 10000)
        model = load_model(argv[2])

        for i in range(dl.number_of_imgs_for_train, dl.number_of_imgs):
            outpath = argv[3] + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            heatmap_path = outpath + "/" + str(dl.imgDataArray[i].img_class) + ".png"

            p_file = Path(heatmap_path)
            if not p_file.exists():  # if segmented does not exists continue...
                print("[ERROR][BIAS METRIC] -> does not exists:", heatmap_path)
                continue
            heatmap = Image.open(heatmap_path)

            tmp = heatmap_path[:-4]
            tmp = tmp[len(argv[3]):]
            tmp = "./dataset_black_bg" + tmp + "_final_masked.jpg"
            print(tmp)

            tmp_file = Path(tmp)
            if not tmp_file.exists():  # if segmented does not exists continue...
                print("[ERROR][BIAS METRIC] -> does not exists:", tmp)
                continue
            mask = Image.open(tmp)

            compute_metric(heatmap, mask)
    if argv[1] == "3":
        dataset_convertor('dataset_black_bg', 'dataset_rand', 'dataset_art')
    if argv[1] == "4":
        directories = next(os.walk(argv[2]))[1]
        directories = sorted(directories)
        i = 0
        for directory in directories:
            for _ in next(os.walk(argv[2] + "/" + directory))[1]:
                i += 1
        print(i, "images processed.")
    if argv[1] == "5":
        dl = DatasetLoader(argv[2], 10000)
        for i in range(0, dl.nb_classes):
            j = 0
            for f in dl.imgDataArray:
                if f.img_class == i:
                    j += 1
            if argv[3] == "csv":
                print(i, ",", j)
            else:
                print("Class", i, "has", j, "shamples")
    if argv[1] == "6":
        import subprocess  # threads, dataset, model
        bashCommand = "1" + argv[2] + argv[3] + argv[4] + "thread" + argv[5]
    if argv[1] == "7":
        create_n_run_mnist(DatasetLoader(argv[2], int(argv[3])), 10)
    if argv[1] == "8":
        dl = DatasetLoader(argv[2], 10000)
        for i in range(0, dl.number_of_imgs):
            try:
                outpath = "101_resized/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
                try:
                    os.makedirs("101_resized/" + dl.imgDataArray[i].directory)
                except OSError:
                    pass
                img = cv2.imread(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                                 dl.imgDataArray[i].name, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (260, 260))
                cv2.imwrite(outpath, img)
            except:
                pass
    if argv[1] == '9':
        files = glob(argv[2] + "/*/*/*")
        for i, file in enumerate(files):
            file_s = file.split('/')
            if file_s[3] != 'resuts.json':
                outpath = file_s[0] + "/" + file_s[1] + "/" + file_s[2] + "/" + file_s[3] + "_colored.jpg"
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_JET)
                cv2.imwrite(outpath, img)


if __name__ == "__main__":
    main()
# 1 4 mnist_png mnist.h5 thread mnist_maps_np
# 1 101_resized caltech.h5 maps_test 1 2 cv2
# 9 maps_test

"""
def create_cam(model, outname, viz_folder, layer_name):

    heatmaps = []
    for path in next(os.walk(viz_folder))[2]:
        # Predict the corresponding class for use in `visualize_saliency`.
        seed_img = utils.load_img(viz_folder + '/' + path, target_size=(256, 256))

        # Here we are asking it to show attention such that prob of `pred_class` is maximized.
        heatmap = img_processing.heatmap_generate.heatmap_generate(seed_img, model, layer_name, None, True)
        heatmaps.append(heatmap)

    cv2.imwrite(outname, utils.stitch_images(heatmaps))


def make_simple_bias_metrics(dataset_name: str, shampeling_rate: int):

    info("[INFO][MAIN]", "Loading...")
    dataset_loader = DatasetLoader(dataset_name, 10000)

    info("[INFO][MAIN]", "Compiling model...")
    vgg16 = VGG16FineTuned(dataset_loader)
    graph_context = tf.get_default_graph()

    bias_metric = BiasMetric(graph_context)
    mc = MonoMetricCallBack(bias_metric=bias_metric,
                            shampleing_rate=shampeling_rate,
                            current_loader=dataset_loader)

    info("[INFO][MAIN]", "Starting training...")
    vgg16.train(10, False, [mc])

    info("[INFO][MAIN]", "Training completed!")"""

"""
def generate_maps_threaded(context, dl: DatasetLoader, model, map_out: str, begining_index: int, end_index: int, number: int):
    with context.as_default():
        # plot CAMs only for the validation data:
        for i in range(begining_index, end_index):
            outpath = map_out + "/" + dl.imgDataArray[i].directory + "/" + dl.imgDataArray[i].name
            try:
                os.makedirs(outpath)
            except OSError:
                continue
            for j in range(0, dl.nb_classes):
                try:
                    outname = outpath + "/" + str(j) + ".tiff"

                    img = cv2.imread(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                                     dl.imgDataArray[i].name, cv2.IMREAD_COLOR)
                    predict_input = np.expand_dims(img, axis=0)
                    predict_input = predict_input.astype('float32')
                    predict_input = preprocess_input(predict_input)
                    predictions = model.predict(predict_input)
                    value = argmax(predictions)
                    start_time = time.time()
                    # input_img, model, class_to_predict, layer_name, image_name=None):
                    heatmap = cam_generate_for_vgg16(
                        input_img=predict_input[0],
                        model=model,
                        class_to_predict=j,
                        layer_name='CAM')
                    Image.fromarray(heatmap).save(outname)
                    print("got cams in", time.time() - start_time)
                    with open(outpath + '/resuts.json', 'w') as outfile:
                        json.dump({'predicted': str(value), "true_label": str(dl.imgDataArray[i].img_class)}, outfile)
                except Exception as e:
                    print("ERROR IN THREAD", number, "error is", e, "redoing...")
                    j -= 1

class MapWorker(Thread):
    def __init__(self, context, dl: DatasetLoader, model, map_out: str, begining_index: int, end_index: int,
                 number: int):
        super().__init__()
        self.context = context
        self.dl = dl
        self.model = model
        self.map_out = map_out
        self.begining_index = begining_index
        self.end_index = end_index
        self.number = number

    def run(self):
        with self.context.as_default():
            print("Thread", self.number, "started...")
            generate_maps_threaded(self.context, self.dl, self.model, self.map_out, self.begining_index, self.end_index,
                          self.number)
"""