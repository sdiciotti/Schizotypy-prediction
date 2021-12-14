# Schizotypy-prediction

*Schizotypy-prediction* is a repository that includes an example of a machine learning predicitive model, [Schizotypy\_group\_prediction.py](Schizotypy\_group\_prediction.py),  and an empty Excel template, [Schizotypy\_rest\_EEG\_template.xlsx](Schizotypy\_rest\_EEG\_template.xlsx), that can be used to organize features extracted from resting state EEG and directly used with the above mentioned code.

If you use this repository, please cite it as:

Trajkovic J, Di Gregorio F, Ferri F, Marzi C, Diciotti S, Romei V. **Resting state alpha oscillatory activity is a valid and reliable marker of schizotypy.** Sci Rep. 2021 May 17;11(1):10379. doi: 10.1038/s41598-021-89690-7. Erratum in: Sci Rep. 2021 Jun 23;11(1):13487. PMID: 34001914; PMCID: PMC8129121.

This document provides a quick introduction to the [Schizotypy\_group\_prediction.py](Schizotypy\_group\_prediction.py)  to help new users get started.  

Please read the [LICENSE.md](./LICENSE.md) file before using *Schizotypy-prediction*.

---------

[Schizotypy\_group\_prediction.py](Schizotypy\_group\_prediction.py) is an example of a machine learning predicitive model, optimized in a nested stratified cross-validation loop repeated 1000 times. By combining alpha activity and connectivity measures ectracted from resting state EEG records, it offers a promising clinical tool, able to identify individuals at-risk of developing psychosis (i.e., high schizotypy individuals).


## Requirements

Open a terminal window (for Unix users) or Anaconda Prompt (for Windows users), activate or create a Python environment with Python version 3.9.1 installed (we recommend to create a new Python environment, see below) and install the following Python packages (if packages already exist, check the version)

```
pip install mlxtend==0.18.0
pip install matplotlib==3.3.3
pip install pandas==1.2.0
pip install numpy==1.19.5
pip install openpyxl==3.0.5
pip install scikit-learn==0.24.0
```


### Create a new local Python virtual environment using conda:
1. Create a new folder with the name of your new environment (e.g., Schizo_env)
2. Open a terminal window (for Unix users) or Anaconda Prompt (for Windows users), from the folder that contains Schizo_env directory and type:

```
conda create --prefix ./Schizo_env
```

```
conda activate ./Schizo_env
```

```
conda install python=3.9.1
```


## Usage

```
python Schizotypy_group_prediction.py --help

Usage: Schizotypy_group_prediction.py [-h] XLSX_file

A machine learning predicitive model, optimized in a nested stratified cross-validation loop repeated 1000 times

positional arguments:
  XLSX_file   XLSX file including data

optional arguments:
  -h, --help  show this help message and exit
```

## Testing
The file [Schizotypy\_rest\_EEG\_template.xlsx](Schizotypy\_rest\_EEG\_template.xlsx) is an empty template with the structure required by [Schizotypy\_group\_prediction.py](Schizotypy\_group\_prediction.py). Fill it with your own features extracted from resting state EEG (details below and in the paper) and try:

```
python Schizotypy_group_prediction.py Schizotypy_rest_EEG_template.xlsx
```

[Schizotypy\_rest\_EEG\_template.xlsx](Schizotypy\_rest\_EEG\_template.xlsx) contains the following columns:

* Schizotypy\_group: label "0" for "Low Schizotypal Group" and "1" for "High Schizotypal Group"
* *AlphaPower\_Frontal\_Left*: maximum power in the alpha range (7-13 Hz) in the left frontal lobe
* *AlphaPower\_Frontal\_Right*: maximum power in the alpha range (7-13 Hz) in the right frontal lobe
* *AlphaPower\_Occipital\_Left*: maximum power in the alpha range (7-13 Hz) in the left occipital lobe
* *AlphaPower\_Occipital\_Right*: maximum power in the alpha range (7-13 Hz) in the right occipital lobe
* *IAF\_Frontal\_Left*: exact frequency in the alpha range (7-13 Hz) containing the maximum power of the left frontal lobe
* *IAF\_Frontal\_Right*: exact frequency in the alpha range (7-13 Hz) containing the maximum power of the right frontal lobe
* *IAF\_Occipital\_Left*: exact frequency in the alpha range (7-13 Hz) containing the maximum power of the left occipital lobe
* *IAF\_Occipital\_Right*: exact frequency in the alpha range (7-13 Hz) containing the maximum power of the right occipital lobe
* *PLI\_Left*: weighted phase lag index as a measure of connectivy between frontal and parieto-occipital lobes in the left hemisphere
* *PLI\_Right*: weighted phase lag index as a measure of connectivy between frontal and parieto-occipital lobes in the right hemisphere
* *TLI\_Left*: time lag index as a measure of communication directionality between frontal and parieto-occipital lobes in the left hemisphere
* *TLI\_Right*: time lag index as a measure of communication directionality between frontal and parieto-occipital lobes in the right hemisphere



## Ouputs
The outputs are stored in two different folders: *Figures* contains the ROC curve plot and *csv\_results* includes all the csv files. Specifically:

* *CombsSelected.csv* reports the number of times each feature combination has been selected
* *Scores.csv* includes average and standard deviation values of balanced accuracy, area under the ROC curve, sensitivity and specificty obtained both in the training and test sets
* *tpr.csv* reports average true positive rate values (among repetitions and nested CV splits)

## Authors
* [**Chiara Marzi**](https://www.unibo.it/sitoweb/chiara.marzi3/en) - *Post-doctoral fellow at Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi", University of Bologna, Bologna, Italy.* Email address: <chiara.marzi3@unibo.it>

* [**Stefano Diciotti**](https://www.unibo.it/sitoweb/stefano.diciotti/en) - *Associate Professor in Biomedical Engineering, Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi", University of Bologna, Bologna, Italy.* Email address: <stefano.diciotti@unibo.it>
