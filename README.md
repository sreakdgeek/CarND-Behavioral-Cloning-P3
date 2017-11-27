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

[image1]: 
[image2]: 
[image3]: 
[image4]: 
[image5]:
[image6]:
[image7]: 
[image8]:
[image9]: 
[image10]:
[image11]:
[image12]:
[image13]:
[image14]:
[image15]:
[image16]:
[image17]:
[image18]:
[image19]:
[image20]:
[image21]:
[image22]:
[image23]:
[image24]:

---
### Data Exploration

Let’s examine few of the track images: 

![alt text][image1] 
![alt text][image2] 
![alt text][image3]

### Observations

### Steering Angle distribution

![alt text][image4]


### Data Pre-processing

Below were the data pre-processing tests done:

### Model Architecture


![alt text][image14] 

Figure: Convolutional Neural Network

### Fine tuning

### Training, Testing and Validation

#### Results

### Training vs Validation Accuracy

Below is the plot for training vs validation accuracy:
