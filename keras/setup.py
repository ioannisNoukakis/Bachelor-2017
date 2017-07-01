from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("img_processing/*.pyx"), requires=['PIL', 'keras', 'quiver_engine', 'sklearn', 'cv2',
                                                             'matplotlib', 'h5py']
)