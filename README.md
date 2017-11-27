**Behavior Cloning** 

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
[image14]:
[image15]:
[image16]:

---
### Data Exploration

Let’s examine few of the track images: 

### Original Central Image

![alt text][image1] 

### After cropping and resizing hte image

![alt text][image2] 

### Left Image (image captured from the left camera)

![alt text][image3] 

### Right Image (image captured from the right camera)

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

![alt text][image5]

About 67% percent of the zero angles were eliminated. Eliminating more zero angles (more than 70%) resulted in poor performance.


Data Augmentation Strategies - Part 1

Using only one run of either of the tracks were sufficient in navigation. Data was augmented
using below strategies:

1. Two runs of Track 1 data
2. Two runs of Track 2 data
3. Recovery from left-side of the road
4. Recovery from right-side of the road
5. Sharp left turn data in Track 1 data
6. Sharp right turn data in Track 2 data


Data Augmentation Strategies - Part 2

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


Figure: Convolutional Neural Network

### Fine tuning

### Training vs Validation Accuracy

Below is the plot for training vs validation accuracy:

![alt text][image12] 

#### Results
