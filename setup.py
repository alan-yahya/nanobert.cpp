from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='custom_extension',
    version='0.1.0',
    description='Custom CUDA extensions for NanoBERT',
    author='Alan Yahya',
    author_email='placeholder@example.com',
    ext_modules=[
        CppExtension(
            name='custom_extension',
            sources=['extension.cpp'],
            extra_compile_args=['/std:c++17'] if os.name == 'nt' else ['-std=c++17']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
    ],
) 