* *ClassID_CNNID.mat* stores the speciesID that used to train caffe models. The first coloumn indicates the ```ClassId``` tagged in the [PlantClef2015 dataset](http://www.imageclef.org/lifeclef/2015/plant) while the second coloumn indicates the speciesID used in the caffe models.

* *author_list.mat* stores the information of the ObservationId and MediaId lists of the plant images captured by all the authors  who contribute to the dataset, details can be found in the [PlantClef2015 official link](http://www.imageclef.org/lifeclef/2015/plant).

* Probability extrated is evaluated via two evaluation metrics: 
    1. the image-centered score```Simg_calculation.m``` 
    2. the observation score ```Sobs_calculation.m```
