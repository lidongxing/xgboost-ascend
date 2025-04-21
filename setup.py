# -*- coding: utf-8 -*-
# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
from distutils import sysconfig

# Get Python paths
PYTHON_INCLUDE = sysconfig.get_python_inc()
PYTHON_LIB = sysconfig.get_config_var('LIBDIR')

# Get Ascend NPU path
ASCEND_HOME = os.environ.get('ASCEND_HOME', '/usr/local/Ascend')
ASCEND_TOOLKIT = os.path.join(ASCEND_HOME, 'ascend-toolkit', 'latest')
ACL_INCLUDE = os.path.join(ASCEND_TOOLKIT, 'acllib', 'include')
ACL_LIB = os.path.join(ASCEND_TOOLKIT, 'acllib', 'lib64')

print(f"PYTHON_INCLUDE: {PYTHON_INCLUDE}")
print(f"PYTHON_LIB: {PYTHON_LIB}")
print(f"ACL_INCLUDE: {ACL_INCLUDE}")
print(f"ACL_LIB: {ACL_LIB}")

class XGBoostBuildExt(build_ext):
    def build_extension(self, ext):
        # Print debug information
        print(f"Include paths: {ext.include_dirs}")
        print(f"Library paths: {ext.library_dirs}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Check if files exist
        acl_h_path = os.path.join(ACL_INCLUDE, 'acl', 'acl.h')
        lib_path = os.path.join(ACL_LIB, 'libascendcl.so')
        python_h_path = os.path.join(PYTHON_INCLUDE, 'Python.h')
        print(f"Checking acl.h: {os.path.exists(acl_h_path)}")
        print(f"Checking libascendcl.so: {os.path.exists(lib_path)}")
        print(f"Checking Python.h: {os.path.exists(python_h_path)}")
        
        # Create build directory if it doesn't exist
        os.makedirs(os.path.join(self.build_lib, 'xgboost_npu'), exist_ok=True)
        
        # Compile C/C++ code
        try:
            cmd = [
                'gcc',
                '-shared',
                '-fPIC',
                '-o', os.path.join(self.build_lib, 'xgboost_npu', 'libxgboost_npu.so'),
                os.path.join('src', 'xgboost_core.c'),
                '-I', ACL_INCLUDE,
                '-I', PYTHON_INCLUDE,
                '-L', ACL_LIB,
                '-L', PYTHON_LIB,
                '-lascendcl',
                f'-lpython{sys.version_info.major}.{sys.version_info.minor}',
                '-O3',
                '-v',
                '-Wl,-rpath,' + ACL_LIB
            ]
            print("Executing command:", ' '.join(cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            print("Compilation stdout:", stdout)
            print("Compilation stderr:", stderr)
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed with error: {e}")
            print(f"Command output: {e.output if hasattr(e, 'output') else 'No output'}")
            print(f"Command stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            raise

# Define extension module
xgboost_extension = Extension(
    name='xgboost_npu.libxgboost_npu',
    sources=['src/xgboost_core.c'],
    include_dirs=[
        ACL_INCLUDE,
        PYTHON_INCLUDE,
    ],
    library_dirs=[
        ACL_LIB,
        PYTHON_LIB,
    ],
    libraries=[
        'ascendcl',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
    ],
    extra_compile_args=['-O3', '-fPIC'],
    extra_link_args=['-shared', f'-Wl,-rpath,{ACL_LIB}']
)

setup(
    name='xgboost-npu',
    version='0.1.0',
    description='XGBoost implementation for Ascend NPU',
    author='lidongxing',
    author_email='lidx@bnu.edu.cn',
    packages=['xgboost_npu'],
    package_dir={'xgboost_npu': 'src'},
    ext_modules=[xgboost_extension],
    cmdclass={'build_ext': XGBoostBuildExt},
    install_requires=[
        'numpy>=1.19.0',
        'mindspore>=2.0.0',
    ],
    python_requires='>=3.7',
)

