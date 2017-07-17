from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("heatmapgenerate.py"), requires=['PIL', 'keras', 'quiver_engine', 'sklearn', 'cv2',
                                                             'matplotlib', 'h5py', 'Cython']
)