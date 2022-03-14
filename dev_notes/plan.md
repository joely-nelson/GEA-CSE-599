## DNAbert

Will use the default model, I will fine tune it for classification.

DNA bert repo: https://github.com/jerryji1993/DNABERT
DNA bert paper: https://sci-hub.st/10.1093/bioinformatics/btab083

### Understanding its structure

### Problem: input sequence length.

https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification

Options that are easy to implement:
 - Take only the first max_len residues of the plasmid, pad if necessary. Likely to fail.
 - Take a random part of the sequence and try to fine tune this way. Very unlikely to work, maybe not worth trying...
 - Break each sequence down into n x max_len residues, then classify each of them and get the average prediction or something like this for classification. I would say even build a classifier for this... Following example of https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f

Example websites that I checked for classifiers on top of bert:
 - https://www.tensorflow.org/text/tutorials/classify_text_with_bert (tensorflow)
 - https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f (BEST)
 - long tests https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f 
 - not checked... https://www.analyticsvidhya.com/blog/2021/06/why-and-how-to-use-bert-for-nlp-text-classification/


Default sequence length for finetunining DNAbert6 (6-mers): 96 6-mers, so this is sequence length of 97.

WAIT!!!! There is an argument about max-seq-length that could make it possible to input longer sequences!!!

I am not sure if finetunining the other models would change sequence length...since kmers are shorters probably it reduces sequence length.

### Things to try:

Might want to try plasmid vs sequences. I think it would complement a bit more and be interesting and easier, and I would have to create a new dataset which is also cool.

Might also want to try swapping dnaBERT 6 to 5 or 4, which changes kMER size of the input.

#### Route 1. Classify with finetuning

The paper and in most cases you start with the pretrained model that understands DNA and then you fine tune it coupling it with a classifier. In the paper example it is a binary classifier. Maybe I could even do engineered plasmid vs wild sequences found in genebank or nature... see if it is effective... this would be a really curious case...
 - Dummy example. Take the two most abundant classes. Just swap the sample data in sample_data/ft/6/train.tsv for the one of my sequences (using the first nucleotides) and one or the other class. See if pretraining script works and I see an increase in accuracy or something like this. What will the sample size be like?
    - In this approach I just need to prepare my input data
    - And then try running their input data.
 - Full example with multiple classes. Will require adapting pretraining code to multiple labels I guess.
 - Full example but not only taking the first 97nts of the sequence but a random part
 - Same as above but instead of a random part, you break down the sequence in multiple subsequences, classify each of them and take the mean or something to make a class prediction or something....or a 1D CNN that aggreagates those features into a single thing..but I don't think it is possible to have it with multiple distances.

#### Route 2. Classifier on top

Strictly, there is no need to finetune the model. One can use the pretrained model (not the finetuned, it would not make sense) and pass in sequences and get an output representation of the sequence. I can then train classifiers on top of this.

 - Use a singular representation. The <cls> token representation after embedding with the model is the one that represents the whole sequence. Then, I can build a classifier on top that learns to match these representations to the class...
    - It can either be using only the first or a random subset of the sequence as input for the model
    - Or a bagged approach where you break down your sequence in multiple subsequences and get representations for each of them and then your model should predict on top of this set of representations.
 - Use the embedded sequence. Not only using the <cls> but everything. I guess it would have higher accuracy but not really interested in it, most approaches building classifiers on top just use the general class that represents all the sequence.


Steps: 
 1. Download DNAbert6
 2. Finetuned classifier with their code. Does it work?
 3. Prepare my input data for the most abundant top 2 classes and swap it for the example finetuning data. Does it work?
 4. Now adapt Finetuned code to multiple classes and use on my data using initial 97nts.
 5. Do the same but for a random part of the sequence.
 6. Same but now aggregating more than one sequence so that one follows the other one...

Not sure if I want to try the classifier on top approach. I guess I would like it but have difficulties thinking about a way to use that classifier... if the output representation of <cls> is a vector then I think I might have a chance bc input is always fixed, if it is just one value... I think it would be impossible for the classifier to have multiple classes. Maybe I could do it with a binary classification problem comparing only the 2 most abundant (and relatively same sizes) samples...


### Refined

The things I tried with binary classes have been very successful.

Next step is going multiclass. 

For this I am going to try 2 things:
 - Try with the top 30 labs. I will find the problem of data unbalance, but I can try balancing out by resampling the sequences that are less abundant or taking a subsample to adjust to the smaller sample size or something in between (that's what I would do) to get a reasonable number of sequences. I can also try totally unbalanced and see how the accuracy drops.
 - Go completely full scale!


