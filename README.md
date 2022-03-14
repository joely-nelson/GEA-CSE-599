# GEA

Installing the environment:

```
conda env create -f environment.yml
```

Getting dnabert:

```
git clone git@github.com:jerryji1993/DNABERT.git dnabert
```

Configuring DNAbert:

```
conda create -n dnabert python=3.6
conda activate dnabert

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

Download pretrained DNABERT6

```
cd dnabert
mkdir model; cd model
mkdir pretrained
```

Download model from [this link](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing). (300Mb)

Keep in pretrained directory, and unzip.

```
unzip 6-new-12w-0.zip
```


TODO: Include links to download the successfully finetuned models.


Adding environments to jupyter notebook

```
conda activate comp_genomics

python -m ipykernel install --user --name=comp_genomics
```
