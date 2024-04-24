# DeepLearnRTX: Deep Learning with Geforce RTX platform

Deep learning with Tensorflow and OpenCV running on Nvidia GPU enabled platform to accelerate model training and inferencing. 

The following steps automated [Tensorflow installation](https://www.tensorflow.org/install/pip#windows-native_1).

## Environment (Windows Native)
1. Install [Visual Studio C++]( 
https://visualstudio.microsoft.com/vs/olderdownloads/) and [CMake](https://microsoft.github.io/AirSim/build_windows).
2. Install [Anaconda](https://www.anaconda.com/products/individual)
3. Setup the [conda](https://www.anaconda.com/) environment:
```
> conda env create -f environment.yml
> conda activate deeplearn-rtx
```
4. Verify environment using pytest
```
> make test
```
5. Deactivate and switch environment
```
> conda deactivate
> conda env list
> conda activate deeplearn-rtx
```
## Execution

