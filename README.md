## Change Detection in Remote Sensing


#### Project Organization

- ```bnlstm```: An implementation of Recurrent Batch Normalization in TensorFlow
- ```config```: Model config and dataset description
- ```data.checkpoint```: Tensorflow model files
- ```data.dataSets```:  Rawdata, in csv format
- ```data.features```:  Temporal data, in Pickle format
- ```data.prediction```: Prediction results
- ```dataloader```: Data iterator
- ```feature```: Functions concern feature preprocessing
- ```model```: Machine learning models: Vgg16, Resnet
- ```utils```: I/O and other utility functions

#### Requirements
- Apache MXNet 1.0
- python 2.7

#### References
1. Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.
2. Cooijmans T, Ballas N, Laurent C, et al. Recurrent batch normalization[J]. arXiv preprint arXiv:1603.09025, 2016.
3. Shahsavari B, Abbeel P. Short-term traffic forecasting: Modeling and learning spatio-temporal relations in transportation networks using graph neural networks[J]. 2015.
4. Della Valle E, Celino I, Dell’Aglio D, et al. Urban Computing: a challenging problem for Semantic Technologies[C]//2nd International Workshop on New Forms of Reasoning for the Semantic Web (NEFORS 2008) co-located with the 3rd Asian Semantic Web Conference (ASWC 2008). 2008.
5. Che Z, Purushotham S, Cho K, et al. Recurrent neural networks for multivariate time series with missing values[J]. arXiv preprint arXiv:1606.01865, 2016.
