Here we try multiclass classification.

However, we use only 30 most abundant classes with weighted random sampling method to balance the number of datapoints in each. So datasets are completely balanced. Sequence length of 101.


### Next up

 - test increasing sequence length to 512
 - test with imbalanced classes
 - test with full dataset

### Estimating run time

Calculating steps: number of sequences in training data / 32 (batch size) * 5 (number of epochs). I think every step takes like 6 seconds?

BUT if you log every 10 steps, every 10 steps you get predictions for the whole EVALUATION datset, which is 1,000 (samples) / 32 (batch size) = 32 * 6sec/it = 3mins just for evaluation!! So it spends waaaay more time evaluating than training it. This was not a problem for the first approach as it was super fast to train but definietly not allowed me to fully train the second. 

Using checkpoints reasonably is a must, specially if not calculating exactly the amount of training required.

7500 training data sequences:

    7500/32*5 = aprox 1170 steps
    if each step is 22secs, this means... 7hours in total

    if I log every 50 steps, I get 23 logs.
    Each log takes aprox 3 mins, so this adds up to 1 hour. It is okay.

    I will make a checkpoint every 200 steps.

    Every epoch is 7500/32=235 samples and it will take about 1h20mins