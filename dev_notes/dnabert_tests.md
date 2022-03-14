# Tests with DNAbert

## 1. Run exactly same pretraining method as in the model

Simply adjust the paths and finetune the model as the authors report in the paper.

```
/home/mexposit/cg/gea/transformers/2_geainit
```

## 2. Use the first residues of the plasmid sequences, try different --max_seq_length parameters

Work done in:

```
/home/mexposit/cg/gea/transformers/2_geainit
```

Basically do the same thing as before but swapping their data by my data using a binary classification problem with the most abundant classes and taking only the first nnn basepairs of the plasmids.

