# xgboost-ascend
a new project for xgboost in ascend NPU310B

sudo apt-get install python3-dev
cd xgboost-npu
gcc -shared -fPIC -o src/libxgboost_npu.so src/xgboost_core.c \
    -I/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include \
    -L/usr/local/Ascend/ascend-toolkit/8.0.0.alpha003/aarch64-linux/lib64 \
    -Wl,-rpath=/usr/local/Ascend/ascend-toolkit/8.0.0.alpha003/aarch64-linux/lib64 \
    -lascendcl \
    $(python3-config --includes)

#python setup.py clean
#python setup.py build
python setup.py install

And then you can import xgboost_npu
