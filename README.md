# Recurrent Neural Networks-based Autoencoders
A PyTorch implementation of [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf)


## Table of Contents:
<!-- Table of contents generated generated by http://tableofcontent.eu -->
- [RecAE-PyTorch](#recae-pytorch)
    - [Project Structure](#project-structure)
    - [Model](#model)
    - [Data](#data)
    - [Requirements](#requirements)
    - [Usage](#usage)


### Project Structure:
The project structure is based on the following [Pytorch Project Template](https://github.com/moemen95/PyTorch-Project-Template)
```
├── agents
|  └── rnn_autoencoder.py # the main training agent for the recurrent NN-based AE
├── graphs
|  └── models
|  |  └── recurrent_autoencoder.py  # recurrent NN-based AE model definition
|  └── losses
|  |  └── MAELoss.py # contains the Mean Absolute Error (MAE) loss
|  |  └── MSELoss.py # contains the Mean Squared Error (MSE) loss
|  |  └── AUCLoss.py # under development (DO NOT USE!)
├── datasets  # contains all dataloaders for the project
|  └── ecg5000.py # dataloader for ECG5000 dataset
├── data
|  └── ECG5000  # contains all ECG time series
├── utils # utilities folder containing metrics, checkpoints and arg parsing (configs).
|  └── assets
|  └── checkpoints.py
|  └── config.py
|  └── metrics.py
|  └── create_config.py
|  └── data_preparation.py
├── notebooks # Folder where adding your notebook
├── experiments # Folder where saving the results of your experiments
├── main.py

```

### Model
#### Encoder

![alt text](./utils/assets/encoder.png "Encoder")


In the encoder each vector <img src="https://render.githubusercontent.com/render/math?math=x^{(t)}"> of a time-window <img src="https://render.githubusercontent.com/render/math?math=x"> of length <img src="https://render.githubusercontent.com/render/math?math=L"> is fed into a recurrent unit to perform the following computation: 

<h1 align='center'> <img src="https://latex.codecogs.com/svg.latex?\large&space;h^{(t)}_{E}=f(x^{(t)},&space;h^{(t-1)}_{E};&space;\theta_{E})" title="\large h^{(t)}_{E}=f(x^{(t-1)},&space;h^{(t-1)}_{E};&space;\theta_{E})" /> </h1>


#### Decoder
![alt text](./utils/assets/decoder.png "Decoder")

In the decoder we reconstruct the time series <img src="https://render.githubusercontent.com/render/math?math=x"> in reverse order: 

<h2 align='center'> <img src="https://latex.codecogs.com/svg.latex?\large&space;h^{(t)}_{D}=f(\hat{x}^{(t&plus;1)},&space;h^{(t&plus;1)}_{D};&space;\theta_{D})" title="\large h^{(t)}_{D}=f(\hat{x}^{(t&plus;1)},&space;h^{(t&plus;1)}_{D};&space;\theta_{D})" /> </h2>


<h3 align='center'><img src="https://latex.codecogs.com/svg.latex?\large&space;\hat{x}^{(t)}&space;=&space;Ah^{(t)}_{D}&plus;b" title="\large \hat{x}^{(t)} = Ah^{(t)}_{D}+b" /> </h3>

### Data

#### Description
The [ECG5000 dataset](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) contains 5000 ElectroCardioGram (ECG) univariate time series of length <a href="https://www.codecogs.com/eqnedit.php?latex=L=5000" target="_blank"><img src="https://latex.codecogs.com/svg.latex?L=5000" title="L=5000" /></a>. Each sequence corresponds to an heartbeat. Five classes are annotated, corresponding to the following labels: Normal (N), R-on-T Premature Ventricular Contraction (R-on-T PVC), Premature Ventricular Contraction (PVC), Supra-ventricular Premature or Ectopic Beat (SP or EB) and Unclassified Beat (UB). For each class we have the number of instances reported in the following Table:

| Class | #Instance |
| --- | --- |
| N | 2919 |
| R-on-T PVC | 1767 |
| PVC | 194 |
| SP or EB | 96 |
| UB | 24 |

Since the main task here is anomaly detection rather than classification, all istances which do not belong to class N have been merged in unique class which will be referred to as Anomalous (AN).

#### Download and data partioning
You can directly download the ECG5000 dataset from [here](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) or by running the script ```utils/data_preparation.py```. This script allows performing data partitioning as well, i.e., splitting your data in training, validation and test set. For more details, run the following: ``` python utils/data_preparation.py -h```


### Requirements
Check [requirements.txt](https://github.com/PyLink88/Recurrent-Autoencoder/blob/main/requirements.txt).

### Usage
- Before running the project, you need to add your configuration into the folder ```configs/``` as found [here](https://github.com/PyLink88/Recurrent-Autoencoder/blob/main/configs/config_rnn_ae.json). To this aim, you can just modify the script ```utils/create_config.py```and then running the following
``` python utils/create_config.py```.
- Finally to run the project: ``` python main.py  configs/config_rnn_ae.json```




# recurrent_ponti