import json
import sys
from glob import glob
from threading import Thread

import PIL
from PIL import Image, ImageOps
from keras.models import load_model


from VGG16_ft import VGG16FineTuned
from bias_metric import BiasWorkerThread
from img_processing import dataset_convertor

from plant_village_custom_model import train_custom_model
from heatmap_generate import *


def main():
    np.random.seed(123)  # for reproducibility
    random.seed(123)
    print("Keras bias v1.0")

    argv = sys.argv
    if argv[1] == "0":
        dataset_convertor('dataset_black_bg', 'dataset_rand', 'dataset_art')
    if argv[1] == "1":
        print("SEED IS", 123)
        vggft = VGG16FineTuned(dataset_loader=DatasetLoader(argv[2], int(argv[6]), force_resize=bool(int(argv[7]))),
                               mode=argv[4])
        vggft.train(int(argv[5]), weights_out=argv[3])
    # ==================================================================================================
    if argv[1] == "2":
        dl = DatasetLoader(argv[2], 10000)
        model = load_model(argv[3])
        print("images to process:", dl.number_of_imgs_for_test)
        generate_maps(dl, model, argv[4], tf.get_default_graph(), all_classes=bool(int(argv[5])),
                      batch_size=int(argv[6]), mode=argv[7])
    if argv[1] == '3':
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
        for j in range(0, number_thread):
            threads[j].join()
    if argv[1] == "4":
        dl = DatasetLoader(argv[2], 10000)
        for i in range(0, dl.nb_classes):
            j = 0
            for f in dl.imgDataArray:
                if f.img_class == i:
                    j += 1
            if argv[3] == "csv":
                print(i, ",", j)
            if argv[3] == "tab":
                print(j, ',')
            else:
                print("Class", i, "has", j, "shamples")
    if argv[1] == "5":
        train_custom_model(DatasetLoader(argv[2], int(argv[3])))
    if argv[1] == '6':
        model = load_model(argv[2])
        create_cam_colored(DatasetLoader('dataset', 10000), model, argv[3] + "_normal.png")
        create_cam_colored(DatasetLoader('dataset_art', 10000), model, argv[3] + "_art.png")
        create_cam_colored(DatasetLoader('dataset_rand', 10000), model, argv[3] + "_rand.png")
        create_cam_colored(DatasetLoader('dataset_black_bg', 10000), model, argv[3] + "_black.png")
    if argv[1] == '7':
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
            except (KeyError, json.decoder.JSONDecodeError):
                total += 1
        if total > 0:
            print('[USER WARNING]', total, 'json files were not correctly formed. Did domething happend during the ' +
                  'first part of this procedure?')
        np.save(argv[3], np.asarray(results_correct_prediction))
        np.save(argv[4], np.asarray(results_wrong_prediction))


if __name__ == "__main__":
    main()
