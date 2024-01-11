from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from utils import *

class Metrics(object):
    def __init__(self, golden_tags, predict_tags, id2task, args):
        # if args.cuda:  
        #     self.golden_tags = list(golden_tags.cpu().numpy())
        # else:
        #     self.golden_tags = list(golden_tags.numpy())    
        self.golden_tags = flatten(golden_tags)
        self.predict_tags = flatten(predict_tags)
        self.id2task = id2task
        
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags() # the True Positive (TPs)
        self.predict_tags_counter = Counter(self.predict_tags) # TP + FP
        self.golden_tags_counter = Counter(self.golden_tags)

        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[self.id2task[tag]] = self.correct_tags_number.get(tag, 0) / (self.predict_tags_counter[tag]+1e-10)
        return precision_scores

    def cal_recall(self):
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[self.id2task[tag]] = self.correct_tags_number.get(tag, 0) / (self.golden_tags_counter[tag]+1e-10)
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[self.id2task[tag]], self.recall_scores[self.id2task[tag]]
            f1_scores[self.id2task[tag]] = 2*p*r / (p+r+1e-10)
        return f1_scores

    def count_correct_tags(self): # computing TP values
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1
        return correct_dict

    def _cal_weighted_average(self):
        weighted_average = {}
        total = len(self.golden_tags)

        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def plot_confusion_matrix(self, true_labels, preds, save_pth, normalize=False, cmap=plt.cm.YlOrRd):
        
        true_labels = flatten(true_labels)
        preds = flatten(preds)        
        cm = confusion_matrix(true_labels, preds)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        plt.rc('font',family='Times New Roman',size='8')
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        #plt.show()
        if not save_pth == None:
            plt.savefig(save_pth, dpi=400) 
            
    def report_scores(self): # printing results
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def return_metrics(self):
        return self.precision_scores, self.recall_scores, self.f1_scores
    
    def return_counts(self):
        return self.correct_tags_number, self.predict_tags_counter, self.golden_tags_counter
        
    
