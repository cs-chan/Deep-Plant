# The GRU-CFA

CLEF 2018 Best of Labs (in Press)


## Description

This is the implementation of our [CLEF 2018](http://clef2018.clef-initiative.eu/) work with titled -- [Plant Classification based on Gated Recurrent Unit](http://cs-chan.com/doc/CLEF2018.pdf). We introduce a Coarse-to-Fined Attention (CFA) module where it can locate the local regions that are highly voted by the GRU method in each plant view.

<img src="CLEF2018.gif" width="50%">

## Citation 
```sh
update soon..
```

## Dependency

* The codes are based on [tensorflow](https://www.tensorflow.org/).

## Dataset
* PlantClef image dataset can be downloaded at [this https URL](http://www.imageclef.org/lifeclef/2015/plant).

* Extract the convolutional features of the E-CNN pretrained using [caffe](http://caffe.berkeleyvision.org/). Details of the E-CNN can be read from [this PDF](http://cs-chan.com/doc/TIP_Plant.pdf). 

	1. After downloaded the PlantClef2015 dataset, users have to categorise the images into their respective species classes based on the information provided at [this https URL](http://www.imageclef.org/lifeclef/2015/plant).
	2. Then, please run the ``` getfeatures.py ``` in the 'CNN' folder to extract the convolutional features from the pretrained [E-CNN](http://www.cs-chan.com/source/DeepPlant/E_CNN.zip) model.
	    Mean file ``` species_mean_aug.npy ``` and the deploy.prototxt file ``` PlantClef_VGGmultipath_deploy  ``` are provided in the 'CNN' folder to run the code.


## Installation and Running

1. Users are required to install [tensorflow](https://www.tensorflow.org/) Library.

2. Users are required to download the necessary files at [this https URL](https://github.com/cs-chan/Deep-Plant/tree/master/PlantStructNet/Dataset) and the aforementioned dataset.

3. Users can train the model from scratch by running the ``` mainClef.py ``` which includes ``` temp_createStruct5.py ``` and ``` attn_7_1_ex.py ```.

4. Users can test the trained RNN model prepared in the [RNN](https://github.com/cs-chan/Deep-Plant/tree/master/GRU-CFA/RNN) folder.

## Quantitative Analaysis

1. Users are required to compute the probability outputs via  ``` compute_prob.py ``` 

2. The probability outputs extrated can be evaluated via the two evaluation metrics mentioned at [this https URL](https://github.com/cs-chan/Deep-Plant/tree/master/HGO-CNN/matlab).

## Attention visualisation

Users can run the ``` visualClef.py ``` for the visualisation of the coarse and fined attention maps. Sample outputs are shown in [CFA Map Samples](https://github.com/cs-chan/Deep-Plant/tree/master/GRU-CFA/CFA%20Map%20Samples) folder.


Note that users are expected to modify the corresponding files to correct path to work properly. Enjoy!


## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to ``` adeline87lee at gmail.com ``` or ``` cs.chan at um.edu.my ```

## Lisense
The project is open source under BSD-3 license (see the ``` LICENSE ``` file). Codes can be used freely only for academic purpose.
