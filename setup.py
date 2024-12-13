from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import multiprocessing

# Determine the number of CPU cores for parallel compilation
num_cores = multiprocessing.cpu_count()

# Set compiler flags for optimization
extra_compile_args = []
if os.name == 'nt':  # Windows
    extra_compile_args.extend([
        '/O2',  # Full optimization
        '/openmp',  # Enable OpenMP
        '/std:c++17',
        '/arch:AVX2',  # Use AVX2 instructions if available
        f'/MP{num_cores}'  # Parallel compilation
    ])
else:  # Linux/Mac
    extra_compile_args.extend([
        '-O3',  # Full optimization
        '-fopenmp',  # Enable OpenMP
        '-std=c++17',
        '-march=native',  # Optimize for current CPU
        f'-j{num_cores}'  # Parallel compilation
    ])

# Configure the extension
extension = CppExtension(
    name='custom_extension',
    sources=['extension.cpp'],
    extra_compile_args=extra_compile_args,
    optional=True  # Continue even if optimization flags aren't supported
)

# Setup the package
setup(
    name='custom_extension',
    version='0.1.0',
    description='Custom CUDA extensions for NanoBERT',
    author='Alan Yahya',
    author_email='placeholder@example.com',
    ext_modules=[extension],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=True,  # Use ninja build system if available
            no_python_abi_suffix=True  # Simpler file naming
        )
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
    ],
) 