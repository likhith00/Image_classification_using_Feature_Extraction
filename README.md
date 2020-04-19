# Finetuning_image_classification

This project is an extension of previous [image-classification](https://github.com/likhith00/image-classification) project. The accuracy gained by the previous project was <b>80.0</b>. In order to improve the accuracy we finetune for our model with pretrained deep learning model.

<b>Fine tuning</b> is a process to take a network model that has already been trained for a given task, and make it perform a second similar task. Assuming the original task is similar to the new task, using a network that has already been designed & trained allows us to take advantage of the feature extraction that happens in the front layers of the network without developing that feature extraction network from scratch.

Fine tuning,

<b>1.</b> Replaces the output layer, originally trained to recognize (in the case of imagenet models) 1,000 classes, with a layer that recognizes the number of classes you require

<b>2.</b> The new output layer that is attached to the model is then trained to take the lower level features from the front of the network and map them to the desired output classes, using SGD

<b>3.</b> Once this has been done, other late layers in the model can be set as 'trainable=True' so that in further SGD epochs their weights can be fine-tuned for the new task too.

In this project , I've performed fine tuning using various deep learning architectures such as <b>VGG16</b>, <b>RESNET50</b>, <b>INCEPTIONV3</b> and <b>XCEPTION</b>. I've used the same hyperparameters and dataset for all architectures in order to compare the model accuracy.

I've uploaded only VGG16 implementation because the other implemetation has exact blue print with exact hyper parameters except The model
To use Resnet50 model,

`from keras.applications.resnet50 import ResNet50`

`resnet = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))` 
                  
Resnet50 takes 224X224 as an input size hence declare `img_height` and `img_width` as <b>224</b>

To use InceptionV3 model,

`from keras.applications.inception_v3 import InceptionV3` 

`inception = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(299, 299, 3))`
                  
 InceptionV3 takes 299X299 as an input size hence declare `img_height` and `img_width` as <b>299</b>
 
 To use Xception model,
 
`from keras.applications.xception import Xception` 

`xception = Xception(include_top=False, weights='imagenet',
                  input_shape=(299, 299, 3))`
                  
 Xception takes 299X299 as an input size hence declare `img_height` and `img_width` as <b>299</b>
 


 <b>Finetuned VGG16 model</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Finetuned resnet50 model</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Finetuned InceptionV3 model</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Finetuned Xceptionmodel</b>
<div class="row">
  <div class="column">
    <img src="/vgg16_images/vgg16_flowchart.png" height="300" width="210">
  </div> 
  <div class="column">
    <img src="/resnet50_images/resnet50_model.png" height="300" width="210">
  </div>
   <div class="column">
    <img src="/inception_images/inceptionV3_model.png" height="300" width="210">
  </div>
     <div class="column">
    <img src="/Xception_images/Xception_model.png" height="300" width="210">
  </div>
</div>


After training, The Accuracy plots of VGG16 are satisfying but inception,xception model plots have lot of fluctuations in the validation accuracy. There is a huge difference of train and validation accuracy of Resnet50. This leads to overfitting and reduction in accuracy.

<b>VGG16 Model Accuracy</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Resnet50 Model Accuracy</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>InceptionV3 Model Accuracy</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Xception Model Accuracy</b>

<div class="row">
  <div class="column">
    <img src="/vgg16_images/vgg16_accuracy.png" height="300" width="210">
  </div> 
  <div class="column">
    <img src="/resnet50_images/resnet50_accuracy.png" height="300" width="210">
  </div>
   <div class="column">
    <img src="/inception_images/inceptionV3_accuracy.png" height="300" width="210">
  </div>
     <div class="column">
    <img src="/Xception_images/Xception_accuracy.png" height="300" width="210">
  </div>
</div>


The Loss plots of resnet50 looks satisfying with a variation between validation accuracy and train accuracy. In other models sometimes the validation accuracy exceeds the train accuracy.This may cause over fitting and hence reduction in accuracy.



<b>VGG16 Model Loss Plot</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Resnet50 Model Loss Plot</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>InceptionV3 Model Loss Plot</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Xception Model Loss Plot</b>

<div class="row">
  <div class="column">
    <img src="/vgg16_images/vgg16_loss.png" height="300" width="210">
  </div> 
  <div class="column">
    <img src="/resnet50_images/resnet50_loss.png" height="300" width="210">
  </div>
   <div class="column">
    <img src="/inception_images/inceptionV3_loss.png" height="300" width="210">
  </div>
     <div class="column">
    <img src="/Xception_images/Xception_loss.png" height="300" width="210">
  </div>
</div>



<b>VGG16 Confusion Matrix</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Resnet50 Confusion Matrix</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>InceptionV3 Confusion Matrix </b>&nbsp;&nbsp;&nbsp;<b>Xception Confusion Matrix</b>


<div class="row">
  <div class="column">
    <img src="/vgg16_images/vgg16_confusion_matrix.png" height="300" width="210">
  </div> 
  <div class="column">
    <img src="/resnet50_images/resnet50_confusion_matrix.png" height="300" width="210">
  </div>
   <div class="column">
    <img src="/inception_images/inceptionV3_confusion_matrix.png" height="300" width="210">
  </div>
     <div class="column">
    <img src="/Xception_images/Xception_confusion_matrix.png" height="300" width="210">
  </div>
</div>

<b>VGG16 Accuracy is 90.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Resnet50 Accuracy is 80.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>InceptionV3 Accuracy is 40.0</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Xception Accuracy is 40.0</b>


From the all Pretrained networks VGG16 has performed well and boosted the accuracy of the model to 90%. Other pretrained networks have not performed well because of overfitting or inefficient hyperparameters. The performance of other models can be improved further building powerful image classification models.
