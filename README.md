# Finetuning_image_classification

This project is an extension of previous [image-classification](https://github.com/likhith00/image-classification) project.The accuracy gained by the previous project is <b>80.0</b>. In order to improve the accuracy we perform finetuning for the our model.

<b>Fine tuning</b> is a process to take a network model that has already been trained for a given task, and make it perform a second similar task. Assuming the original task is similar to the new task, using a network that has already been designed & trained allows us to take advantage of the feature extraction that happens in the front layers of the network without developing that feature extraction network from scratch.

Fine tuning,

<b>1.</b> Replaces the output layer, originally trained to recognize (in the case of imagenet models) 1,000 classes, with a layer that recognizes the number of classes you require

<b>2.</b> The new output layer that is attached to the model is then trained to take the lower level features from the front of the network and map them to the desired output classes, using SGD

<b>3.</b> Once this has been done, other late layers in the model can be set as 'trainable=True' so that in further SGD epochs their weights can be fine-tuned for the new task too.

In this project , I've performed fine tuning using various deep learning architectures such as <b>VGG16</b>, <b>RESNET50</b>, <b>INCEPTION</b> and <b>EXCEPTION</b>. I've used the same hyperparameters and dataset for all architectures in order to compare the model accuracy.
