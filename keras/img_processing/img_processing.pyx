from PIL import Image
import numpy
import os

from scipy.misc import fromimage

from random_art_img_generator import Art
import scipy
import scipy.cluster


def dataset_convertor(dataset_directory, outfolder_random, outfolder_art):
    print("converting dataset...")
    directories = next(os.walk(dataset_directory))[1]
    for directory in directories:
        for i, file_name in enumerate(next(os.walk(dataset_directory + "/" + directory))[2]):
            image_splitter(Image.open(dataset_directory + "/" + directory + "/" + file_name, "r"), file_name,
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

    if not os.path.isdir(outfolder_random + "/" + classe):
        os.makedirs(outfolder_random + "/" + classe)
    if not os.path.isdir(outfolder_art + "/" + classe):
        os.makedirs(outfolder_art + "/" + classe)

    background.paste(foreground, (0, 0), foreground)
    background.save(outfolder_random + "/" + classe + "/" + "rand_" + filename, "JPEG")

    background2.paste(foreground, (0, 0), foreground)
    background2.save(outfolder_art + "/" + classe + "/" + "art_" + filename, "JPEG")


def filter_img(img, new_img, f):
    """
    Puts transparency pixel everytime a pixel that is in f range condition is met.
    
    :param img: the image 
    :param new_img: the image where the filtered image will be stored
    :param f: the filter function. f(Int) => Boolean.
    :return: 
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
    :return: the new image
    """
    mask = mask.convert("RGBA")

    mask1 = Image.new("RGBA", image.size)
    mask2 = Image.new("RGBA", image.size)

    filter_img(mask, mask1, lambda x: x < 10)
    filter_img(mask, mask2, lambda x: x != 0)

    img1 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask1)
    img2 = Image.composite(image.convert("RGBA"), Image.new("RGBA", image.size), mask2)

    filter_img(img1, img1, lambda x: x < 10)
    filter_img(img2, img2, lambda x: x < 10)

    return img1, img2


def most_dominant_color(image):
    """
    from : https://gist.github.com/samuelclay/918751
    
    Find the most dominant color in an image.
    
    :param image: the image
    :return: the most dominant color
    """

    NUM_CLUSTERS = 15

    # Convert image into array of values for each point.
    ar = fromimage(image)
    shape = ar.shape

    # Reshape array of values to merge color bands.
    if len(shape) > 2:
        ar = ar.reshape(scipy.product(shape[:2]), shape[2])

    # convert to float
    ar = ar.astype('float32')

    # Get NUM_CLUSTERS worth of centroids.
    codes, _ = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    # Pare centroids, removing blacks and whites and shades of really dark and really light.
    original_codes = codes
    for low, hi in [(60, 200), (35, 230), (10, 250)]:
        codes = scipy.array([code for code in codes
                             if not ((code[0] < low and code[1] < low and code[2] < low) or
                                     (code[0] > hi and code[1] > hi and code[2] > hi))])
        if not len(codes):
            codes = original_codes
        else:
            break

    # Assign codes (vector quantization). Each vector is compared to the centroids
    # and assigned the nearest one.
    vecs, _ = scipy.cluster.vq.vq(ar, codes)

    # Count occurences of each clustered vector.
    counts, bins = scipy.histogram(vecs, len(codes))

    # Find the most frequent color, based on the counts.
    index_max = scipy.argmax(counts)
    return codes[index_max][:3]


def color_distance(c1, c2):
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2
    return numpy.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def main():

    im1, im2 = merge_images_mask(
        Image.open('./dataset/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510.JPG'),
        Image.open('./segmentedDB/Apple___Apple_scab/fcd4d0fd-30c9-4b05-b0ea-ca74fd3cad72___FREC_Scab 3510_final_masked.jpg')
    )

    print(most_dominant_color(im1))
    # dataset_convertor("./segmentedDB", "./dataset_rand", "./dataset_art")


if __name__ == "__main__":
    main()
