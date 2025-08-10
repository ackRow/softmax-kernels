from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.3.0"

# Define the C++ extension modules
ext_modules = [
    CUDAExtension(
        name='cuda_softmax_kernel',
        sources=[
            'binding.cpp',
            'kernel_article1.cu',
            'kernel_article2.cu',
            'kernel_article3.cu',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-DNDEBUG',
                '-ffast-math',
                '-march=native',
                '-funroll-loops',
                '-std=c++17'
            ],
            'nvcc': [
                '-O3',
                '-use_fast_math',
                '-lineinfo',
                '-Xptxas=-v',
                '-maxrregcount=128',
                '-arch=sm_75'
            ]
        }
    )
]

setup(
    name="cuda_softmax_kernel",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)