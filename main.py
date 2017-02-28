from utility import *
import numpy as ny


def main():

    with Image.open("test.jpg") as jpgFile:
        filtre = ny.matrix([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        print("Size of image: ", jpgFile.size)
        i = ImageFilter(jpgFile)
        i.applyFilterOnImage(filtre, 4)

if __name__ == "__main__":
    main()
