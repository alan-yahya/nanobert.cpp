import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import multiprocessing

def build_extension():
    # Determine the number of CPU cores for parallel compilation
    num_cores = multiprocessing.cpu_count()
    
    # Set compiler flags for optimization
    extra_compile_args = []
    if sys.platform == 'win32':
        extra_compile_args.extend([
            '/O2',  # Full optimization
            '/openmp',  # Enable OpenMP
            '/std:c++17',
            '/arch:AVX2',  # Use AVX2 instructions if available
            f'/MP{num_cores}'  # Parallel compilation
        ])
    else:
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

if __name__ == "__main__":
    from setuptools import setup
    # Set compiler flags for optimization
    extra_compile_args = []
    if sys.platform == 'win32':
        extra_compile_args.extend([
            '/O2',  # Full optimization
            '/openmp',  # Enable OpenMP
            '/std:c++17',
            '/arch:AVX2',  # Use AVX2 instructions if available
            f'/MP{num_cores}'  # Parallel compilation
        ])
    else:
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
    setup()

    # Build in parallel
    setup(
        name='custom_extension',
        ext_modules=[extension],
        cmdclass={
            'build_ext': BuildExtension.with_options(
                use_ninja=True,  # Use ninja build system if available
                no_python_abi_suffix=True  # Simpler file naming
            )
        },
    )

if __name__ == "__main__":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "All"  # Build for all CUDA architectures
    
    # Enable parallel build in distutils
    if sys.platform == 'win32':
        os.environ['CL'] = '/MP'
    
    try:
        build_extension()
    except Exception as e:
        print(f"Error during build: {str(e)}")
        # Try fallback build without optimizations
        print("Attempting fallback build...")
        setup(
            name='custom_extension',
            ext_modules=[CppExtension(
                name='custom_extension',
                sources=['extension.cpp'],
                extra_compile_args=['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']
            )],
            cmdclass={'build_ext': BuildExtension}
        )
