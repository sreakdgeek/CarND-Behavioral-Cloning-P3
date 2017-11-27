## Behavior Cloning

### Introduction


Behavior Cloning project requires to clone user's driving behavior on simulated training tracks and use the track image data to predict the steering angle. Cloned driver is able to use the predicted steering angle to navigate through the testing tracks. There are two track provided for the training track - Basic track and jungle track. Data from both the tracks were used for training. Below are the steps performed to clone the driving behavior:

*	Exploratory Data Analysis
*	Data Pre-processing
*	Data Augmentation
*	Build Convolutitional Neural Network Architecture
*	Train, validate, test and fine tune model and Network architecture
*	Use the trained model to predict the steering angle and navigate through the tracks

[//]: # (Image References)

[image1]: ./images/center_image_before_cropping.JPG "Center img before cropping"
[image2]: ./images/center_image_after_cropping.JPG "Center img after cropping"
[image3]: ./images/left_image.JPG "Left Image"
[image4]: ./images/right_image.JPG "Right Image"
[image5]: ./images/distribution_before.JPG "Distribution Before"
[image6]: ./images/distribution_after.JPG "Distribution After"
[image7]: ./images/transform_flip_before.JPG "Before Flip"
[image8]: ./images/transform_flip_after.JPG "After Flip"
[image9]: ./images/transform_translate_before.JPG "Before Translate"
[image10]: ./images/transform_translate_after.JPG "After Translate"
[image11]: ./images/transform_before_incr_contrast.JPG "Before contrast increase"
[image12]: ./images/transform_after_incr_contrast.JPG "After contrast increase"
[image13]: ./images/Evaluation.JPG "Evaluation - Train vs Validation"
[image14]: ./images/model_summary_1.JPG "Model summary"
[image15]: ./images/model_summary_2.JPG "Model summary"
[image16]: ./images/model_summary_3.JPG "Model summary"

---
### Data Exploration

Let’s examine few of the track images: 

#### Original Central Image

![alt text][image1] 

#### After cropping and resizing the image

![alt text][image2] 

#### Left Image (image captured from the left camera)

![alt text][image3] 

#### Right Image (image captured from the right camera)

![alt text][image4]

Approach followed in this project is losely based on NVIDIA's End to End Learning for Self Driving Cars paper:

https://arxiv.org/pdf/1604.07316.pdf

### Data Pre-processing

Below were the data pre-processing steps followed:

1. Crop the image from top and bottom to remove parts such as sky, car hood, which will not be very useful in prediction.
2. Eliminate zero-degree steering angles to balance the distribution and thus reduce bias
3. Data augmentation - Capture data from multiple runs of track 1, track 2, recovery scenarios from left and right, etc
4. Data augmentation - Transformations - Random flip, translate, increase contrast

### Steering Angle distribution

Observing the steering angle distribution, tells us that the distribution is skewed as the majority of the steering angles are zero (or straight line driving). This could cause a bias in navigation as the simulated car may not steer to the left or right and take sharp turns. Eliminating the training images with zero steering angle may help in coming up with the more balanced distribution of the steering angles.

![alt text][image5]

After eliminating zero angles, below was the distribtuion of steering angles:

![alt text][image6]

About 67% percent of the zero angles were eliminated. Eliminating more zero angles (more than 70%) resulted in poor performance.


### Data Augmentation Strategies - Part 1

Using only one run of either of the tracks were sufficient in navigation. Data was augmented
using below strategies:

1. Two runs of Track 1 data
2. Two runs of Track 2 data
3. Recovery from left-side of the road
4. Recovery from right-side of the road
5. Sharp left turn data in Track 1 data
6. Sharp right turn data in Track 2 data


### Data Augmentation Strategies - Part 2

Further training data was augment with transformations on the images. About 25% of the image
data was randomly chosen and transformed by randomly flipping, translating or increasing the
contrast on left, right and center images.

#### Before Flip

![alt text][image7] 

#### After Flip

![alt text][image8] 

#### Before translation

![alt text][image9] 

#### After translation

![alt text][image10] 

#### Before increasing contrast

![alt text][image11] 

#### After increasing contrast

![alt text][image12] 

### Model Architecture

I have losely followed NVIDIA's CNN architecture. My model only varies in terms of number of filters and number of neurons in
FC layers and dropouts added. Below are the layer descriptions:

    1) Preprocessing Layer - Image normalization
    2) Convolution Layer # 1 - 32 filters, 5x5, RELU activation, Max pool with stride 2, Dropout (20%)
    3) Convolution Layer # 2 - 32 filters, 5x5, RELU activation, Max pool with stride 2, Dropout (20%)
    4) Convolution Layer # 3 - 48 filters, 5x5, RELU activation, Max pool with stride 2, Dropout (20%)
    5) Convolution Layer # 4 - 64 filters, 5x5, RELU activation, Max pool with stride 2, Dropout (20%)
    6) Convolution Layer # 5 - 64 filters, 5x5, RELU activation, Max pool with stride 2, Dropout (20%)
    7) Flatten layer
    8) FC Layer - 1 - 1024 (neurons)
    9) FC Layer - 2 - 128 (neurons)
    10) FC Layer - 3 - 64 (neurons)
    11) Output Layer - 1 neuron (no activation) - Steering Angle Prediction

#### Model Summary

Below is keras model output summary:

![alt text][image14] 
![alt text][image15] 
![alt text][image16] 

### Fine tuning

I have not exactly followed the precise network architecture of NVIDIA's paper. My implementation had different number of filters
and number of neurons in FC layer. I have used RELU activation function instead of ELU though I have experimented with ELU. Model
check points was a very useful strategy to save the model after each epoch. Validation loss was not particulary useful metric
to predict the usefulness of the model.

### Training vs Validation Accuracy

Below is the plot for training vs validation accuracy:

![alt text][image13] 

#### Results

Link to youtube video - Track 1 only (model 1): https://youtu.be/1DqVjwtNQyQ

Link to youtube video - Track 1 only (model 2): https://youtu.be/cQSi2x5Cdok

### Improvements

Currently my model is partially successful on track 2 and needs more fine-tuning of the model architecture as well as add more recovery 
scenarios. Also since the steering angle decisions not only depend on where the car was in current time frame but also on where the car was on previous time frame, LSTM could be used to model the sequential nature of steering decisions. Also if we can detect the lane lines it helps to adjust steering angles in relation to distnace from the lane lines. Further, a reinforcement learning framework where a reward for driving within the lane lines and penalty for crossing the lane lines could help define a model which may not require much training data. State of the system is defined by the pixel intensities and actions are defined by the steering angle, breaking possition, throttle etc. Objective would to select actions in a given state such that reward (safe driving) is maximized.
