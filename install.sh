# apt install build-essential g++ gcc libeigen3-dev ffmpeg libsm6 libxext6  -y

conda create --name drag
conda activate drag
conda install -c conda-forge python=3.10.11 scipy==1.10.1 numpy==1.24.3 boost==1.80.0 flann==1.9.1 cmake==3.25.0 tqdm
pip install pygccxml==2.2.1 castxml==0.4.5 pyplusplus==1.8.5 pybullet==3.2.5 opencv-python==4.7.0.72 shapely==2.0.1 matplotlib

conda install spot
git clone https://github.com/ompl/ompl.git
cd ompl
git fetch origin 21213304bfe7eda5dac5cae6ce1dded45490f2de
git checkout 21213304bfe7eda5dac5cae6ce1dded45490f2de

# wget -O - https://github.com/ompl/ompl/archive/1.6.0.tar.gz | tar zxf -
# cd ompl-1.6.0

mkdir -p build/Release
cd build/Release
# replace path
# cmake --fresh -DCMAKE_INSTALL_PREFIX=$HOME/remp_ros/ompl-1.6.0 -DPYTHON_EXECUTABLE=/home/hwpeng/.conda/envs/drag310/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..
cmake --fresh -DCMAKE_INSTALL_PREFIX=$HOME/remp_ros/ompl -DPYTHON_EXECUTABLE=/home/hwpeng/.conda/envs/drag310/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..

# cmake -DCMAKE_INSTALL_PREFIX=/root/ompl-1.6.0 -DPYTHON_EXEC=/root/miniforge3/envs/drag/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..
make -j 16 update_bindings
make -j 16
sudo make install



'''bash 
'==========================================python 3.8.19============================================================='
# apt install build-essential g++ gcc libeigen3-dev ffmpeg libsm6 libxext6  -y

conda create --name drag_ros38
conda activate drag_ros38
conda install -c conda-forge python=3.8.19 scipy==1.10.1 numpy==1.24.3 boost==1.80.0 flann==1.9.1 cmake==3.25.0 tqdm
pip install pygccxml==2.2.1 castxml==0.4.5 pyplusplus==1.8.5 pybullet==3.2.5 opencv-python==4.7.0.72 shapely==2.0.1 matplotlib

conda install spot
git clone https://github.com/ompl/ompl.git
cd ompl
git fetch origin 21213304bfe7eda5dac5cae6ce1dded45490f2de
git checkout 21213304bfe7eda5dac5cae6ce1dded45490f2de

# wget -O - https://github.com/ompl/ompl/archive/1.6.0.tar.gz | tar zxf -
# cd ompl-1.6.0

mkdir -p build/Release
cd build/Release
# replace path
# cmake --fresh -DCMAKE_INSTALL_PREFIX=$HOME/remp_ros/ompl-1.6.0 -DPYTHON_EXECUTABLE=/home/hwpeng/.conda/envs/drag310/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..
# cmake -DCMAKE_INSTALL_PREFIX=/root/ompl-1.6.0 -DPYTHON_EXEC=/root/miniforge3/envs/drag/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..
cmake --fresh -DCMAKE_INSTALL_PREFIX=$HOME/ompl -DPYTHON_EXECUTABLE=/home/hwpeng/.conda/envs/drag_ros38/bin/python -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ../..

make -j 16 update_bindings
make -j 16
sudo make install

# 报错 1  
#File "/home/hwpeng/remp_ros38/mcts/search.py", line 853, in MCTS
# def _select_and_expand(self) -> tuple[Node, bool]:
# TypeError: 'type' object is not subscriptable
nano /home/hwpeng/remp_ros38/mcts/search.py
#python文件开头加入
from typing import Tuple as tuple

#报错 2
#FileNotFoundError: [Errno 2] No such file or directory: 'logs/temp-0.obj'
remp_ros38目录下，添加文件夹/remp_ros38/logs

'''
