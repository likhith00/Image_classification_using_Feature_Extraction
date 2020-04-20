# Finetuning_image_classification

This project is an extension of previous [image-classification](https://github.com/likhith00/image-classification) project. The accuracy gained by the previous project was <b>80.0</b>. In order to improve the accuracy we finetune for our model with pretrained deep learning model.

<b>Fine tuning</b> is a process to take a network model that has already been trained for a given task, and make it perform a second similar task. Assuming the original task is similar to the new task, using a network that has already been designed & trained allows us to take advantage of the feature extraction that happens in the front layers of the network without developing that feature extraction network from scratch.

Fine tuning,

<b>1.</b> Replaces the output layer, originally trained to recognize (in the case of imagenet models) 1,000 classes, with a layer that recognizes the number of classes you require

<b>2.</b> The new output layer that is attached to the model is then trained to take the lower level features from the front of the network and map them to the desired output classes, using SGD

<b>3.</b> Once this has been done, other late layers in the model can be set as 'trainable=True' so that in further SGD epochs their weights can be fine-tuned for the new task too.

In this project , I've performed fine tuning using various deep learning architectures such as <b>VGG16</b>, <b>RESNET50</b>, <b>INCEPTIONV3</b> and <b>XCEPTION</b>. I've used the same hyperparameters and dataset for all architectures in order to compare the model accuracy.

I've uploaded only [VGG16](https://neurohive.io/en/popular-networks/vgg16/) implementation because the other implemetation has exact blue print with exact hyper parameters except The model

To use [Resnet50](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624) model,

`from keras.applications.resnet50 import ResNet50`

`resnet = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))` 
                  
<b>Resnet50</b> takes 224X224 as an input size hence declare `img_height` and `img_width` as <b>224</b>

To use [InceptionV3](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) model,

`from keras.applications.inception_v3 import InceptionV3` 

`inception = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(299, 299, 3))`
                  
 <b>InceptionV3</b> takes 299X299 as an input size hence declare `img_height` and `img_width` as <b>299</b>
 
 To use [Xception](https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568) model,
 
`from keras.applications.xception import Xception` 

`xception = Xception(include_top=False, weights='imagenet',
                  input_shape=(299, 299, 3))`
                  
 <b>Xception</b> takes 299X299 as an input size hence declare `img_height` and `img_width` as <b>299</b>
 

<img src="/final_images/model_final.PNG">

After training, The Accuracy plots of VGG16 are satisfying but inception,xception model plots have lot of fluctuations in the validation accuracy. There is a huge difference of train and validation accuracy of Resnet50. This leads to overfitting and reduction in accuracy.

<img src="/final_images/accuracy_final.PNG">



The Loss plots of resnet50 looks satisfying with a variation between validation accuracy and train accuracy. In other models sometimes the validation accuracy exceeds the train accuracy.This may cause over fitting and hence reduction in accuracy.

<img src="/final_images/loss_final.PNG">

To evaluate the model performance analyse the confusion matrix 

<img src="/final_images/confusion_matrix_final.PNG">


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>VGG16 Accuracy is 90.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Resnet50 Accuracy is 80.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>InceptionV3 Accuracy is 40.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Xception Accuracy is 40.0</b>


From the all Pretrained networks VGG16 has performed well and boosted the accuracy of the model to 90%. Other pretrained networks have not performed well because of overfitting or inefficient hyperparameters. The performance of other models can be improved further building powerful image classification models.
