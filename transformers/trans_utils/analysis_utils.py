import pandas as pd

def read_scores_from_eval(file_path, include_top10acc=False, step_freq=None):
    """
    To identify which score is which I have to read lines 464-468 of the run_finetune.py script,
    which basically sorts the scores and I see that the order is the same at which they are printed in the logs.
    So by looking at the logs I can be sure that the order at which I assign them is correct
    
    The way each score is computed can also be found in:
    `/home/mexposit/cg/gea/dnabert/src/transformers/data/metrics/__init__.py`
    at the `acc_f1_mcc_auc_pre_rec` function
    
    Arguments:
     - file_path: path to the eval_results.txt file in the models
     - include_top10acc: set to true if eval_results.txt also reports the top10 accuracy
     
    Outputs:
     - dataframe with the scores
    """
    scores = []

    with open(file_path) as f:
        for line in f:
            line_sc = line.strip().split(' ')
            sc_dic = {
                'acc': float(line_sc[1]),
                'auc': float(line_sc[2]),
                'f1': float(line_sc[3]),
                'mcc': float(line_sc[4]),
                'precision': float(line_sc[5]),
                'recall': float(line_sc[6])
            }
            if include_top10acc:
                sc_dic['top10acc'] = float(line_sc[7])

            scores.append(sc_dic)
            
    scores = pd.DataFrame.from_records(scores)
    # optionally add ['step'] indicating the step from each evaluation
    if step_freq is not None:
        scores['step'] = [i*step_freq for i in range(1, len(scores)+1)]
    return scores


def rev_comp(seq):
    revcomp = ''
    for nt in seq:
        if nt.upper() == 'A':
            revcomp = revcomp+'T'
        elif nt.upper() == 'G':
            revcomp = revcomp+'C'
        elif nt.upper() == 'C':
            revcomp = revcomp+'G'
        elif nt.upper() == 'T':
            revcomp = revcomp+'A'
        else:
            raise ValueError(f'Unrecognized nucleotide {nt}')
    return revcomp[::-1]