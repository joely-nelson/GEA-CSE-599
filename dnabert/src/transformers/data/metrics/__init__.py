# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    # MOD add top10 metric
    def top10_accuracy(labels, probs):
        """
        Calculate top 10 accuracy (lab of origin within the 10 most likely classes)
        From: https://www.drivendata.co/blog/genetic-attribution-benchmark/
        """
        # get the indices for top 10 predictions for each row; these are the last ten in each row
        # Note: We use argpartition, which is O(n), vs argsort, which uses the quicksort algorithm 
        # by default and is O(n^2) in the worst case. We can do this because we only need the top ten
        # partitioned, not in sorted order.
        top10_idx = np.argpartition(probs, -10, axis=1)[:, -10:]
    
        # index into the classes list using the top ten indices to get the class names
        # top10_preds = estimator.classes_[top10_idx]
        # here the classes are labeled using integers so they should match
        top10_preds = top10_idx

        # check if y-true is in top 10 for each set of predictions
        mask = top10_preds == labels.reshape((labels.size, 1))
        
        # take the mean
        top_10_accuracy = mask.any(axis=1).mean()
        return top_10_accuracy

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
    
    def acc_f1_mcc(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(labels, preds)
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc
        }

    def acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "aupr": aupr,
            "precision": precision,
            "recall": recall,
        }

    def acc_f1_mcc_auc_pre_rec(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds, average="macro")
        recall = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
        return {
            "acc": acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
        }

    # MOD, add top10 accuracy score
    def acc_f1_mcc_auc_pre_rec_top10(preds, labels, probs):
        acc = simple_accuracy(preds, labels)
        precision = precision_score(y_true=labels, y_pred=preds, average="macro")
        recall = recall_score(y_true=labels, y_pred=preds, average="macro")
        f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
        mcc = matthews_corrcoef(labels, preds)
        auc = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
        top10acc = top10_accuracy(labels, probs)
        return {
            "acc": acc,
            "top10acc": top10acc,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "top10acc": top10acc,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels, probs=None):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name in ["dna690", "dnapair"]:
            return acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs)
        elif task_name == "dnaprom":
            return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
            # return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "dnasplice":
            return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
        # MOD, include specific task and top10 accuracy
        elif task_name == "dnagea30":
            return acc_f1_mcc_auc_pre_rec_top10(preds, labels, probs)
        # MOD, include specific task and top10 accuracy
        elif task_name == "dnageaall":
            return acc_f1_mcc_auc_pre_rec_top10(preds, labels, probs)
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
