from PIL import Image, ImageOps
import numpy
import os
from random_art_img_generator import Art


def dataset_convertor(dataset_directory, outfolder_random, outfolder_art):
    print("converting dataset...")
    directories = next(os.walk(dataset_directory))[1]
    for directory in directories:
        for i, file_name in enumerate(next(os.walk(dataset_directory + "/" + directory))[2]):
            image_splitter(Image.open(dataset_directory + "/" + directory + "/" + file_name, file_name),
                           outfolder_random, outfolder_art, directory)
            print("Treated", file_name, "successfully.")


def image_splitter(foreground, filename, outfolder_random, outfolder_art, classe):
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
    background.save(outfolder_random + "/" + classe + "/" + "rand_" + filename, "JPEG")

    background2.paste(foreground, (0, 0), foreground)
    background2.save(outfolder_art + "/" + classe + "/" + "art_" + filename, "JPEG")


def filter(mask, maskNew, f):
    datas = mask.getdata()
    new_data = []
    for item in datas:
        if f(item[0]) and f(item[1]) and f(item[2]):
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    maskNew.putdata(new_data)


def merge_images_mask(image, mask):
    mask = mask.convert("RGBA")

    mask1 = Image.new("RGBA", image.size)
    mask2 = Image.new("RGBA", image.size)

    filter(mask, mask1, lambda x: x < 10)
    filter(mask, mask2, lambda x: x != 0)

    img1 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask1)
    img2 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask2)
    return img1, img2


def main():
    im1, im2 = merge_images_mask(
        Image.open('./dataset/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510.JPG'),
        Image.open('./segmentedDB/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510_final_masked.jpg')
    )

    im1.show()
    im2.show()
    # dataset_convertor("./segmentedDB", "./dataset_rand", "./dataset_art")


if __name__ == "__main__":
    main()
