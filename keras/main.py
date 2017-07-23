import json
import sys
from glob import glob
from threading import Thread

from PIL import Image, ImageOps
from keras.models import load_model
from scipy.misc import toimage

from VGG16_ft import VGG16FineTuned
from bias_metric import compute_metric, compute_bias
from img_processing import dataset_convertor
from mnist_model import create_n_run_mnist
from model_utils import get_outputs_generator, reduce_opacity
from plant_village_custom_model import *
from numpy import argmax
from keras.applications.imagenet_utils import preprocess_input
import time
import pyximport;

pyximport.install()
from heatmapgenerate import *


# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-1
# http://cnnlocalization.csail.mit.edu/
# https://arxiv.org/pdf/1312.4400.pdf
# https://www.quora.com/What-is-global-average-pooling
# https://arxiv.org/pdf/1512.04150.pdf
# https://arxiv.org/pdf/1512.03385.pdf
# http://lcn.epfl.ch/tutorial/english/perceptron/html/learning.html
# https://github.com/fchollet/keras/issues/4446


class BiasWorkerThread(Thread):
    def __init__(self, a, b, base_d, files_path):
        Thread.__init__(self)
        self.a = a
        self.b = b
        self.files_path = files_path
        self.base_d = base_d

    def run(self):
        for i in range(self.a, self.b):
            try:
                start_time = time.time()
                with open(self.files_path[i]) as data_file:
                    data = json.load(data_file)
                if data['predicted'] == data['true_label']:
                    score = compute_bias(self.base_d, self.files_path[i], data['predicted'])
                    if score == -1:
                        continue
                    score_n01 = compute_bias(self.base_d, self.files_path[i], data['predicted'], 'normalizer01')
                    score_nmin = compute_bias(self.base_d, self.files_path[i], data['predicted'], 'normalizerMin')
                    with open(self.files_path[i], 'w') as outfile:
                        json.dump({'predicted': data['predicted'], "true_label": data['true_label'],
                                   'score': score, 'score_n01': score_n01, 'score_nmin': score_nmin}, outfile)
                else:
                    score_predicted = compute_bias(self.base_d, self.files_path[i], data['predicted'])
                    if score_predicted == -1:
                        continue
                    score_predicted_n01 = compute_bias(self.base_d, self.files_path[i], data['predicted'],
                                                       'normalizer01')
                    score_predicted_nmin = compute_bias(self.base_d, self.files_path[i], data['predicted'],
                                                        'normalizerMin')

                    score_true_label = compute_bias(self.base_d, self.files_path[i], data['true_label'])
                    score_true_label_n01 = compute_bias(self.base_d, self.files_path[i], data['true_label'],
                                                        'normalizer01')
                    score_true_label_nmin = compute_bias(self.base_d, self.files_path[i], data['true_label'],
                                                         'normalizerMin')

                    with open(self.files_path[i], 'w') as outfile:
                        json.dump({'predicted': data['predicted'], "true_label": data['true_label'],
                                   'score_predicted': score_predicted, 'score_predicted_n01': score_predicted_n01,
                                   'score_predicted_nmin': score_predicted_nmin,
                                   'score_true_label': score_true_label, 'score_true_label_n01': score_true_label_n01,
                                   'score_true_label_nmin': score_true_label_nmin},
                                  outfile)
            except json.decoder.JSONDecodeError:
                print('[USER WARNING]', 'Json was malformed. Pehaps you cam generation was interrupted?')
            print("ok(", time.time() - start_time, ") seconds")


def create_cam(dl: DatasetLoader, model, outname: str, im_width=256):
    os.makedirs(outname)
    heatmaps = []
    for i in range(0, dl.number_of_imgs):
        predict_input = (cv2.imread(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                                    dl.imgDataArray[i].name, cv2.IMREAD_COLOR))
        base = Image.open(dl.baseDirectory + "/" + dl.imgDataArray[i].directory + "/" +
                          dl.imgDataArray[i].name)
        predict_input = predict_input.astype('float32')
        predict_input = np.expand_dims(predict_input, axis=0)
        predict_input = preprocess_input(predict_input)

        output_generator = get_outputs_generator(model, 'CAM')
        layer_outputs = output_generator(predict_input)[0]

        inputs = model.input
        output_predict = model.get_layer('W').output
        fn_predict = K.function([inputs], [output_predict])
        prediction = fn_predict([predict_input])[0]
        value = np.argmax(prediction)

        w = model.get_layer("W").get_weights()[0]
        heatmap = cv2.resize(layer_outputs[:, :, 0], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
        heatmap *= w[0][value]
        for z in range(1, layer_outputs.shape[2]):  # Iterate through the number of kernels
            img = cv2.resize(layer_outputs[:, :, z], (im_width, im_width), interpolation=cv2.INTER_CUBIC)
            heatmap += img * w[z][value]

        heatmap = toimage(heatmap)
        heatmap = cv2.applyColorMap(np.uint8(np.asarray(ImageOps.invert(heatmap))), cv2.COLORMAP_JET)
        # heatmap = cv2.putText(heatmap, dl.imgDataArray[i].name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        heatmap = toimage(heatmap)
        heatmap = reduce_opacity(heatmap, 0.5)
        base.paste(heatmap, (0, 0), heatmap)
        base.save(outname + "/" + dl.imgDataArray[i].name)
        # base.show()
        # heatmaps.append(np.asarray(ImageOps.invert(base)))

        # cv2.imwrite(outname, utils.stitch_images(heatmaps))


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
    if argv[1] == "1":
        # 1 101_resized caltech.h5 maps_test_tf 1 2 tf
        dl = DatasetLoader(argv[2], 10000)
        model = load_model(argv[3])
        print("images to process:", dl.number_of_imgs_for_test)
        generate_maps(dl, model, argv[4], tf.get_default_graph(), all_classes=bool(int(argv[5])),
                      batch_size=int(argv[6]), mode=argv[7])

    if argv[1] == '2':
        files_path = glob(argv[2] + "/*/*/*.json")
        number_of_files_to_process = len(files_path)
        print(number_of_files_to_process, "images to process")
        number_thread = int(argv[3])
        inc = int(number_of_files_to_process / number_thread)
        a = 0
        b = inc
        threads = []
        print(number_thread, "worker will be used and each will process", inc, "images")
        for i in range(0, number_thread):
            t = BiasWorkerThread(a, b, argv[2], files_path)
            threads.append(t)
            t.start()
            print('thread', i, 'will take care of', a, 'to', b)
            a = b
            b += inc
        for t in threads:
            t.join()
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
    if argv[1] == '10':
        model = load_model(argv[2])
        create_cam(DatasetLoader('visual', 10000), model, argv[3] + "_normal", 260)
        create_cam(DatasetLoader('visual_art', 10000), model, argv[3] + "_art", 260)
        create_cam(DatasetLoader('visual_rand', 10000), model, argv[3] + "_rand", 260)
    if argv[1] == '11':
        results_correct_prediction = []
        results_wrong_prediction = []
        files_path = glob(argv[2] + "/*/*/*.json")
        total = 0

        for filep in files_path:
            try:
                with open(filep) as data_file:
                    data = json.load(data_file)
                if data['predicted'] == data['true_label']:
                    a = [('predicted', data['predicted']),
                         ('true_label', data['true_label']),
                         ('score', data['score']),
                         ('score_n01', data['score_n01']),
                         ('score_nmin', data['score_nmin'])]
                    results_correct_prediction.append(np.asarray(a))
                else:
                    a = [('predicted', data['predicted']),
                         ('true_label', data['true_label']),
                         ('score_predicted', data['score_predicted']),
                         ('score_predicted_n01', data['score_predicted_n01']),
                         ('score_predicted_nmin', data['score_predicted_nmin']),
                         ('score_true_label', data['score_true_label']),
                         ('score_true_label_n01', data['score_true_label_n01']),
                         ('score_true_label_nmin', data['score_true_label_nmin'])]
                    results_wrong_prediction.append(np.asarray(a))
            except json.decoder.JSONDecodeError:
                total += 1
        if total > 0:
            print('[USER WARNING]', total, 'json files were not correctly formed. Did domething happend during the ' +
                  'first part of this procedure?')
        np.save('results_correct_prediction', np.asarray(results_correct_prediction))
        np.save('results_wrong_prediction', np.asarray(results_wrong_prediction))
    if argv[1] == '12':
        files_path = glob(argv[2] + "/*/*/*.json")
        total = 0

        # for experiements. Add cams by class and total
        cams_total_pre_class = np.zeros((3, 256, 256))
        for file_p in files_path:
            try:
                with open(file_p) as data_file:
                    data = json.load(data_file)
                if data['predicted'] != data['true_label']:
                    splitted = file_p.split('/')
                    img_path_true_label = 'dataset_black_bg/' + splitted[-3] + '/' + splitted[-2] + '/' + data['true_label'] + '.tiff'

                    cam_true_label = cv2.imread(img_path_true_label, cv2.IMREAD_UNCHANGED)
                    cams_total_pre_class[int(data['true_label'])] += cam_true_label

            except json.decoder.JSONDecodeError:
                total += 1
        if total > 0:
            print('[USER WARNING]', total, 'json files were not correctly formed. Did domething happend during the ' +
                  'first part of this procedure?')
        np.save('cams_total_pre_class', np.asarray(cams_total_pre_class))

if __name__ == "__main__":
    main()
# 1 4 mnist_png mnist.h5 thread mnist_maps_np
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
