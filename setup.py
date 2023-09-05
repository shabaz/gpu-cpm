import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy


### This file is adapted from https://github.com/rmcgibbo/npcuda-example.

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    #home = "/usr/local/cuda-11.2"
    #nvcc = "/usr/local/cuda-11.2/bin/nvcc"

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    # try fallback if lib64 dir does not exist (e.g. on conda)
    if not os.path.exists( cudaconfig['lib64'] ):
       cudaconfig['lib64'] = pjoin( home, 'lib' ) 

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension('gpucpm',
                sources=['C_src/python_wrapper.cpp', 'A_src/cpm.cu', 'A_src/kernels.cu', 'B_link.cu'],
                library_dirs=[CUDA['lib64']],
                libraries=['cudart', 'cudadevrt'],
                runtime_library_dirs=[CUDA['lib64']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                        #'-arch=sm_70', 
                extra_compile_args={'gcc': [],
                    'nvcc': ['-std=c++17',
                    #'-ccbin=gcc-7', # this can help if cuda complains about a too new C compiler
                    '--gpu-architecture=compute_70', '--gpu-code=sm_70',
                        '-rdc=true','-use_fast_math', '--compiler-options', "'-fPIC'", "--ptxas-options=-v", "-maxrregcount=40"],
                                    'nvcclink': [
                                        #'-L/usr/lib/x86_64-linux-gnu', # this can help when there are linker errors in the final step
                                        '--gpu-architecture=compute_70', '--gpu-code=sm_70',
                                        '--device-link', '--compiler-options', "'-fPIC'"]
                                    },
                include_dirs = [numpy_include, CUDA['include'], 'src'])




def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    self.cuda_object_files = []

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print( "compile call on " + src + " ::: " + obj )
        src = src[2:]
        if src == 'link.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcclink']
            src = self.cuda_object_files[0]
            cc_args = self.cuda_object_files[1:]
        elif os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
            self.cuda_object_files.append(obj)
        else:
            postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


if __name__ == "__main__": setup(name='gpucpm',
      # metadata is now defined in pyproject.toml mostly.

      # this is necessary so that the swigged python file gets picked up
      py_modules=['gpucpm'],

      ext_modules = [ext],

      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},

      # since the package has c code, the egg cannot be zipped
      zip_safe=False)
