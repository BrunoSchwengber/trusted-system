# An Online System for Unsupervised Botnet Detection (TRUSTED)

SCHWENGBER, Bruno Henrique; LIMA, Michele Nogueira. Um Sistema para Detecção Não Supervisionada e Online de Botnets. In: Anais do XIX Simpósio Brasileiro em Segurança da Informação e de Sistemas Computacionais. SBC, 2020.

## Requirements

Install all python libraries in requirements.txt

```
pip install requirements.txt
``` 


## Code

The system has 6 parameters and follow the instructions given below.

**Parameters:**
* 1º input dataset
* 2º window size
* 3º new data percentage
* 4º threshold for AUC
* 5º input dataset labels
* 6º clustering distance

**Command line instructions:**

* python trusted.py dataset_name window_size new_data_percentage AUC_threshold clustering_distance
* sample: 
```
python trusted.py datasets/botnet_2014.csv 100 0.1 0.7 botnet_2014labels.csv 0.1
```

**The code will output:** 
* Final accuracy, precision, recall and f-score
* Total elapsed time (from beginning of the stream to the end)
* A accuracy plot (dividing data stream into 30 chunks)
* A file containing the evaluation final results

# Datasets

* Botnet 2014 
```
https://www.unb.ca/cic/datasets/botnet.html
```
* CTU-13
```
https://www.stratosphereips.org/datasets-ctu13
```
