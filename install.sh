# apt install build-essential g++ gcc libeigen3-dev ffmpeg libsm6 libxext6  -y

conda create --name drag
conda activate drag
conda install -c conda-forge python=3.10.11 scipy==1.10.1 numpy==1.24.3 boost==1.80.0 flann==1.9.1 cmake==3.25.0 tqdm
pip install pygccxml==2.2.1 castxml==0.4.5 pyplusplus==1.8.5 pybullet==3.2.5 opencv-python==4.7.0.72 shapely==2.0.1 matplotlib

wget -O - https://github.com/ompl/ompl/archive/1.6.0.tar.gz | tar zxf -
cd ompl-1.6.0
mkdir -p build/Release
cd build/Release
# replace path
cmake --fresh -DCMAKE_INSTALL_PREFIX=/root/ompl-1.6.0 \
             -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
             -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
             -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
             ../..
# cmake -DCMAKE_INSTALL_PREFIX=/root/ompl-1.6.0 -DPYTHON_EXEC=/root/miniforge3/envs/drag/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..
make -j 16 update_bindings
make -j 16
make install
