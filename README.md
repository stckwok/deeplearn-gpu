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

1. Show usages for training with ASL dataset
```
> python code\asl.py --help
```
2. Kick off model training with (default=CNN model) and 10 epocs (default=20)
```
> python code\asl.py -n 10
```
3. Perform inference with new images, never seen by the trained model and evaluate its performance
```
> python code\asl_predict.py
```

## Model Evaluation

![screenshot](results\Using_CNN_ASL.png)

Model Evaluation
 - loss: 0.1137 - accuracy: 0.9587
[INFO] Accuracy: 95.87%

![screenshot](results\Model_Evaluation_20epocs.png)

Model Summary

![screenshot](results\Model_Summary.png)

## Prediction

Make prediction using the model saved from previous steps. 

![screenshot](results\Prediction.png)