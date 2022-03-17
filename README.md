# Genetic Engineering Attribution modeling

By Joely Nelson and Marc Exposit Goy.

This repository contains the code for the CNN and transformer classifiers implemented to identify the lab-of-origin of the Addgene plasmid dataset.

For more information, check the [final report](GEA_final_report.pdf)

## Repository contents:

 - [`cnn`](cnn/): Contains all code used to build and analyze the CNN classifiers.
 - [`transformers`](transformers/): Contains all code used to fine-tune the transformer models.
 - [`data`](data/): Contains the input data of the Addgene dataset. See [readme](data/readme.md) inside the directory.
 - [`dnabert`](dnabert/): Contains the [dnabert](https://github.com/jerryji1993/DNABERT) module adapted to multiclass classification, report more outputs, and calculate top 10 accuracy.
 - [`dev_notes`](dev_notes/): Contains some notes intended for the developers / drafts.

## CNN approach

The CNN and transformer approaches use different environments. This is because transformers have to use the environment in the `dnabert` module and the CNN uses a more recent python version.

Start by creating the environment.

 - If running Windows, create the exact same environment that was used (with specific binaries for windows): `conda env create -f cnn/envir_windows.yml` and then activate it `conda activate comp_genomics`.
 - If running another platform, use an equivalent environment: `conda env create -f cnn/environment.yml`

Now, after making sure you placed the data in [`data`](data/) directory, you can run the notebooks [Simple CNN Models.ipynb](cnn/Simple%20CNN%20Models.ipynb) and [Simple.ipynb](cnn/Simple.ipynb) to recreate the results. Note they both use functions from the [utils](cnn/utils.py) file.

## Transformer approach
### Download model weights

In order to use the transformer, you first need to download the [dnabert](https://github.com/jerryji1993/DNABERT) module and model weights. 
We modified the original module to multiclass classification specifically for this problem, added some functions to report the probabilities and predictions after each test evaluation step, and added the Top10 accuracy metric. Hence, we have included the download of this module in this repository inside the [`dnabert`](dnabert/) directory.

However, you still need to download the weights of the model, which are too large to be kept in github.

If downloading the model from scratch on the original `dnabert` module, you would need to create the following directories to place the unzziped model inside `pretrained`. However, since in this case we already made this folders you can skip this step.

```
cd dnabert
mkdir model; cd model
mkdir pretrained
```

Then, you need to download the model weights from [this link](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing) (300Mb).

After placing the zip file in the `pretrained` directory you can unzip the file with your favorite program or running `unzip 6-new-12w-0.zip`. Done!

Note that if you want to experiment more with `dnabert` you can clone it from the [original repository](https://github.com/jerryji1993/DNABERT) running `git clone git@github.com:jerryji1993/DNABERT.git dnabert` and check out its github repo for the documentation and existing issues.

### Creating the environment

Transformer models were trained using the same `dnabert` environment provided in the original module. To create this environment, run the following commands:

```
conda create -n dnabert python=3.6
conda activate dnabert

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

python3 -m pip install --editable .
cd dnabert/examples
python3 -m pip install -r requirements.txt
```

Note that you may want to install jupyter notebook on top of this by running `conda activate dnabert; conda install jupyterlab`. If you need to include the environment to jupyter notebook to use it as a kernel, you can activate the environment first and then run `python -m ipykernel install --user --name=comp_genomics`.

### Looking at the fine-tuning and analysis scripts

The `dnabert` model was modified in situ in the `dnabert` module. By looking at the commit history it is possible to see what we changed in relation to the original module (first commit).

The scripts used for data processing, model fine-tuning and results analysis are inside the `transformers` directory. Each of the subdirectories contains a jupyter notebook to process the data input, the processed data divided in training `train.csv` and test `dev.csv` files, directories to log the checkpoints, accuracies and predictions at every evaluation step, and a bash script that can be run to fine-tune the model.
 - [1_pretraintest](transformers/1_pretraintest): First test to run the initial example the authors of DNABERT provide for fine-tuning, not using the GEA data.
 - [2_geainit](transformers/2_geainit): Binary classifier model with 100nts as input sequence taking the beggining of the plasmid sequence.
 - [3_binmulti](transformers/3_binmulti): Binary classifier model with 100nts as input sequence taking 8 random subsequences of every plasmid.
 - [4_geathirty](transformers/4_geathirty): Multiclass model with 30 classes and a randomly selected sequence input of 100nts of length.
 - [5_gea30long](transformers/5_gea30long): Multiclass model with 30 classes and a randomly selected sequence input of 512nts of length.
 - [6_gea30xl](transformers/6_gea30xl): Multiclass model with 30 classes and a randomly selected sequence input of 1024nts of length.
 - [7_gea500](transformers/7_gea500): Multiclass model using 512nts of input sequence length and a training set with sequences from all labels in the full dataset (1,314 classes).
 - [analysis](transformers/analysis): Jupyter notebooks to analyze the accuracy of the models on the test sets during training and compare them.
    - `1_binaryclass.ipynb`: Notebook to compare the binary classifier models in `2_geainit` and `3_binmulti`.
    - `2_multi30class.ipynb`: Notebook to compare the 30 label classifier models in `4_geathirty`, `5_gea30long` and `6_gea30xl`.
    - `3_alldata.ipynb`: Notebook to analyze the model trained with sequences from all dataset labels in `7_gea500`.
 - [trans_utils](transformers/trans_utils): Contains auxiliary scripts.
    - `analysis_utils.py`: Functions to load metrics results on the test set during training and reverse complement DNA sequences.
    - `run_finetune_mod.py`: Script used to run fine-tuning with support for multiclass, top10 accuracy, and logging the predictions and probabilities at each evaluation step in `numpy` arrays.

Note that the fine-tuned models were not included in this repository, but they could be obtained again by running the bash scripts for fine-tuning in a proper system.
