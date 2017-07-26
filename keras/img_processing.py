from PIL import Image, ImageEnhance
import numpy
import os

from random_art_img_generator import Art


def dataset_convertor(dataset_directory, outfolder_random, outfolder_art):
    """
    Convert a dataset by creating two new datasets.
    One with a random background and one with an 'art' background.

    :param dataset_directory:
    :param outfolder_random:
    :param outfolder_art:
    :return:
    """
    print("converting dataset...")
    directories = next(os.walk(dataset_directory))[1]
    for directory in directories:
        for i, file_name in enumerate(next(os.walk(dataset_directory + "/" + directory))[2]):
            image_splitter(Image.open(dataset_directory + "/" + directory + "/" + file_name, "r"), file_name,
                           outfolder_random, outfolder_art, directory)
            print("converted", file_name, "successfully.")


def image_splitter(foreground, filename, outfolder_random, outfolder_art, the_class):
    """
    Take an image an generate two new image.
    One with a random background and one with an 'art' background.
    
    :param foreground: The image to be treated.
    :param filename: The name of the image to be treated.
    :param outfolder_random: the path to the outfolder of the random dataset.
    :param outfolder_art: the path to the outfolder of the art dataset.
    :param the_class: The name of the class of the dataset.
    :return: -
    """
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

    if not os.path.isdir(outfolder_random + "/" + the_class):
        os.makedirs(outfolder_random + "/" + the_class)
    if not os.path.isdir(outfolder_art + "/" + the_class):
        os.makedirs(outfolder_art + "/" + the_class)

    background.paste(foreground, (0, 0), foreground)
    new_name = filename[:-17]
    background.save(outfolder_random + "/" + the_class + "/" + new_name + "jpg", "JPEG")

    background2.paste(foreground, (0, 0), foreground)
    background2.save(outfolder_art + "/" + the_class + "/" + new_name + 'jpg', "JPEG")


def filter_img(img, new_img, f):
    """
    Puts transparency pixel every time a pixel that is in f range condition is met.
    
    :param img: the image 
    :param new_img: the image where the filtered image will be stored
    :param f: the filter function. f(Int) => Boolean.
    :return: -
    """

    datas = img.getdata()
    new_data = []
    for item in datas:
        if f(item[0]) and f(item[1]) and f(item[2]):
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    new_img.putdata(new_data)


def merge_images_mask(image, mask):
    """
    Merge an image with it's mask so it return only the allowed part by the mask.
    
    :param image: the image
    :param mask: the mask
    :return: the new images
    """
    mask = mask.resize((224, 224))
    mask1 = Image.new("RGBA", image.size)
    mask2 = Image.new("RGBA", image.size)

    filter_img(mask, mask1, lambda x: x < 5)
    filter_img(mask, mask2, lambda x: x != 0)

    img1 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask1)
    img2 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask2)

    filter_img(img1, img1, lambda x: x < 5)
    filter_img(img2, img2, lambda x: x < 5)

    return img1, img2


def reduce_opacity(im, opacity):
    """
    Returns an image with reduced opacity.
    Taken from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/362879
    """
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def main():
    im1, im2 = merge_images_mask(
        Image.open('./dataset/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510.JPG'),
        Image.open(
            './dataset_black_bg/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510_final_masked.jpg')
    )

    im1.show()
    im2.show()


if __name__ == "__main__":
    main()
