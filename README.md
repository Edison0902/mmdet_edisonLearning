# mmdet_edisonLearning
Learning use for mmdetection

## NOTES:
1. gcc 4.9
2. g++ 4.9
3. ubuntu 18.04
4. python 3.6
5. NVIDIA driver stay latest version
6. cuda 9.0 

------------------------------------------------------------
## INSTALLATION:
1. Install cython
2. Install pytorch
3. Install mmcv
        git clone https://github.com/open-mmlab/mmcv.git
        cd mmcv
        pip install .
4. Install mmdet
        cd mmdetection
        ./compile.sh
        python setup.py install
5. Install cocoapi