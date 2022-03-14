Same as previous:
 - 30 most abundant, balanced in all sets by undersampling
 - here went over the 512 sequence length up to 2048, so it breaks it down in 4 pieces of 512 and then concatenates their representations for the final classifier.


### Next up

 - test with imbalanced classes
 - test with full dataset
 - use embedded sequence representation and train classifier on top

### Estimating run time

Calculating steps: number of sequences in training data / 32 (batch size) * 5 (number of epochs). Every step is 160secs because they are longer... and I guess it could be even longer since each sequence will have 4 inferences.

