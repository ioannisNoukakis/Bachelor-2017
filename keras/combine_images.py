# Combine multiple images into one.
#
# To install the Pillow module on Mac OS X:
#
# $ xcode-select --install
# $ brew install libtiff libjpeg webp little-cms2
# $ pip install Pillow
#

from __future__ import print_function
import os
from glob import glob
from sys import argv

from PIL import Image

files = glob(argv[1] + "/*")

n = 6
s = 256
result = Image.new("RGB", (n*s, (len(files)//n+1)*s))

for index, file in enumerate(files):
    path = os.path.expanduser(file)
    img = Image.open(path)
    x = index % n * 256
    y = index // n * 256
    w, h = img.size
    print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
    result.paste(img, (x, y, x + w, y + h))

result.save(argv[2])
