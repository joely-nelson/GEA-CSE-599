
## Results from the first day

Their pretrained model trains really fast and gets up to high accuracy easily (of course, it is using their data).

I tested binary classification, so taking only the second two most abundant classes that have balanced number of plasmids. And I followed two different approaches.

### First approach

Take the first 100 nucleotides of each plasmid sequence, randomly reverse translate it. Dataset has about 2200 sequences for each of the two classes for training, and 500 of each of the two classes in the evaluation dataset. 

### Second approach

For each sequence in the dataset, take 8 random stretches of 100 nucleotides and randomly reverse translate them with prob 50%. So suddenly our data has 40,000 sequences. Take 1,000 for evaluation. Doing only one epoch already takes for ever and the accuracy is super high, so there is no need to augment the data like this!

This one I left it for 24hours, and it completed around 2,000 steps. 2,000steps * 32 samples/step = 64,000 sequences, so it is about one epoch. But the accuracy was super high even before completing that epoch. Kept checkpoints every 500 steps so I can use them.

### Extrapolations from these models

I think evaluating them on 1,000 sequences is not a lot. I propose a way to compare them. Take the initial sequences from these datasets, get random 100 nucleotide stretches (with random reverse translate) and use the model to predict them. Do it upon 10,000 sequences or so... will see if the high accuracy is still maintained... WARNING, maybe I should make sure that those selected 10,000 random sequences were not accidentally included in the training data!

### Training conclusions

Logging every 10 steps is not great when you want to make so many steps.

Calculating steps: number of sequences in training data / 32 (batch size) * 5 (number of epochs). I think every step takes like 6 seconds?

BUT if you log every 10 steps, every 10 steps you get predictions for the whole EVALUATION datset, which is 1,000 (samples) / 32 (batch size) = 32 * 6sec/it = 3mins just for evaluation!! So it spends waaaay more time evaluating than training it. This was not a problem for the first approach as it was super fast to train but definietly not allowed me to fully train the second. 

Using checkpoints reasonably is a must, specially if not calculating exactly the amount of training required.



## TODO

Look into weighted averaged sampling of the samples as a way to balance them

DNAbert works for sequences up to 512 nucleotides out of the box, so no longer restricted to it.

There seem to be some flags to get even longer sequences ready to work.



An alternative approach:
 - use pretrained model and they have an example where they load the model, use it to featurize a sample. They say outputs[0] is the embedded representation of the input DNA sequence of length about 700, there is also outputs[1] that is the [CLS] token. But recent studies show it is better to use mean(outputs[0]) than just the [CLS] token. So I can use this script to use the nonpretrained model to get embeddings for everysequence and then build convolutions or whatever on top to build a multiclass classifier. Maybe I could build a 1D CNN.



## TODO INFERENCE SCRIPT

Get a python script or notebook that can be used to load the corresponding model and make correct inference for a given sequence or set of sequences.