from PIL import Image
import numpy
import os
from random_art_img_generator import Art


def simple_load_images(dataset_directory, outfolder):
    print("converting dataset...")
    directories = next(os.walk(dataset_directory))[1]
    for directory in directories:
        for i, file_name in enumerate(next(os.walk(dataset_directory + "/" + directory))[2]):
            merge_images(dataset_directory + "/" + directory + "/" + file_name, file_name, outfolder, directory)
            print("Treated", file_name, "successfully.")


def merge_images(filepath, filename, outfolder, classe):
    size = 256, 256

    foreground = Image.open(filepath)
    imarray = numpy.random.rand(256, 256, 3) * 255
    background = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    background2 = Art().redraw()
    foreground = foreground.convert("RGBA")
    datas = foreground.getdata()

    new_data = []
    for item in datas:
        if item[0] < 10 and item[1] < 10 and item[2] < 10:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    foreground.putdata(new_data)
    background.paste(foreground, (0, 0), foreground)
    background.save(outfolder + "/" + classe + "/" + "rand_" + filename, "JPEG")

    background2.paste(foreground, (0, 0), foreground)
    background2.save(outfolder + "/" + classe + "/" + "art_" + filename, "JPEG")


def main():
    simple_load_images("./segmentedDB", "./datasetNoBiais")


if __name__ == "__main__":
    main()
