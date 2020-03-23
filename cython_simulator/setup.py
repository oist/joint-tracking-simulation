from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("evolve",
                             ["evolve.pyx"],
                             language='c++',
                             include_dirs=[numpy.get_include()]),
                   Extension("agents",
                             ["agents.pyx"],
                             language='c++',
                             include_dirs=[numpy.get_include()]),
                   Extension("CTRNN",
                             ["CTRNN.pyx"],
                             language='c++',
                             include_dirs=[numpy.get_include()]),
                   Extension("simulate",
                             ["simulate.pyx"],
                             language='c++',
                             include_dirs=[numpy.get_include()])],
      )

# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize("evolve.pyx")
# )
#

# from distutils.core import setup
# from Cython.Build import cythonize
#ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
#setup(
#    ext_modules = cythonize("CTRNN.pyx", **ext_options)
#)

# ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
# setup(
#     ext_modules = cythonize("*.pyx", **ext_options)
# )


# python3 setup.py build_ext --inplace
# %run -p main_joint.py 'buttons' 124

