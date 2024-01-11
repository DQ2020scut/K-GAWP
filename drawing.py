import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn
import os
import xlrd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
from itertools import cycle

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

def process_xls_rows(temp_rows):
    temp_hits = []
    for _, row in enumerate(temp_rows):
        #print(row)
        if any([t.ctype==1 for t in row]):
            continue
        else:
            temp_hits.append({'hit6': row[0].value, 'hit3':row[1].value, 'hit1':row[2].value})
    return temp_hits[:-1] # the last metric is the avg result


def process_xls_sheets(xls_pth, model):
    xls_file = xlrd.open_workbook(xls_pth)

    sp_df = data_transfer(process_xls_rows(xls_file.sheets()[0]), model+'-SP-G') 
    nosp_df = data_transfer(process_xls_rows(xls_file.sheets()[1]), model+'-NSP-G')
    
    sp_detail_df = data_transfer(process_xls_rows(xls_file.sheets()[2]), model+'-SP-D')
    nosp_detail_df = data_transfer(process_xls_rows(xls_file.sheets()[3]), model+'-NSP-D')
    
    model_df = pd.concat([sp_df, nosp_df])
    model_df_detail = pd.concat([sp_detail_df, nosp_detail_df])
    return model_df, model_df_detail


def add_pd(hits, hit_type, model):
    df = pd.DataFrame(columns=['h_values', 'model', 'hit'])
    for i, hit in enumerate(hits):
        df.loc[i] = [hit, model, hit_type]
        #df = df.append({'value':hit, 'model':model, 'hit':hit_type}, ignore_index=True)
    return df


def data_transfer(model_hits, model):
    df = pd.DataFrame(columns=['h_values', 'model', 'hit'])
    hit6 = []
    hit3 = []
    hit1 = []
    for _, data in enumerate(model_hits):
        hit6.append(data['hit6'])
        hit3.append(data['hit3'])
        hit1.append(data['hit1'])
    
    df6 = add_pd(hit6, 'hit@6', model)
    df3 = add_pd(hit3, 'hit@3', model)
    df1 = add_pd(hit1, 'hit@1', model)
    #df = pd.concat([df6, df3, df1])
    df = df.append(df6, ignore_index=True)
    df = df.append(df3, ignore_index=True)
    df = df.append(df1, ignore_index=True)
    return df


def box_plot_figure(df, figure_out_path, fig_size=(15, 10), detail=False):
    figsize = fig_size
    #fig, _ = plt.subplots()
    ax = seaborn.boxenplot(x='model', y='h_values', data=df, hue='hit', orient='v', linewidth=0.8, palette=seaborn.color_palette('deep', 10))
    
    plt.xlabel('models', fontsize=12)
    plt.ylabel('hit@N values', fontsize=12) 
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(title='metrics', fontsize=12)
    
    if not detail:
        ax.axhline(y=0.743, c=flatui[0], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit6 sp
        ax.axhline(y=0.711, c=flatui[1], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit3 sp
        ax.axhline(y=0.572, c=flatui[2], ls='-', linewidth=1.2, zorder=0, clip_on=False) #hit1 sp
        ax.axhline(y=0.570, c=flatui[3], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit6 nsp
        ax.axhline(y=0.344, c=flatui[4], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit3 nsp
        ax.axhline(y=0.165, c=flatui[5], ls='--', linewidth=1.2, zorder=0, clip_on=False) #hit1 nsp
    else:
        ax.axhline(y=0.826, c=flatui[0], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.707, c=flatui[1], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.617, c=flatui[2], ls='-', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.601, c=flatui[3], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.429, c=flatui[4], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        ax.axhline(y=0.208, c=flatui[5], ls='--', linewidth=1.2, zorder=0, clip_on=False)
        
    plt.savefig(figure_out_path, dpi=300)
    #plt.show()    


def loss_plot(xls_file, sheet_id, save_name=None, figsize=(7,5)):
    LSTM=False
    if sheet_id<2:
        LSTM=True
    
    if LSTM==True:
        loss_df = pd.DataFrame(columns=['Step', 'LSTM-SP', 'LSTM-NSP'])
    else:
        loss_df = pd.DataFrame(columns=['Step', 'DNN-SP', 'BERT-SP', 'DNN-NSP', 'BERT-NSP'])
    
    for i, row in enumerate(xls_file.sheets()[sheet_id]):
        if any([t.ctype==1 for t in row]):
            continue
        else:
            if LSTM==True:
                loss_df.loc[i] = [i, row[0].value, row[1].value]
            else:
                loss_df.loc[i] = [i, row[0].value, row[1].value, row[2].value, row[3].value]
    steps = list(loss_df['Step'].values)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)
    if LSTM==True:
        plt.plot(steps, loss_df['LSTM-SP'], flatui[0], label='LSTM-SP', linewidth=1.5)
        plt.plot(steps, loss_df['LSTM-NSP'], flatui[3], label='LSTM-NSP', linewidth=1.5)
    else:
        plt.plot(steps, loss_df['DNN-SP'], flatui[1], label='DNN-SP', linewidth=1.5)
        plt.plot(steps, loss_df['BERT-SP'], flatui[2], label='BERT-SP', linewidth=1.5)
        plt.plot(steps, loss_df['DNN-NSP'], flatui[4], label='DNN-NSP', linewidth=1.5)
        plt.plot(steps, loss_df['BERT-NSP'], flatui[5], label='BERT-NSP', linewidth=1.5)
        
    plt.legend(fontsize=12, loc='upper right')
    plt.xticks(np.linspace(0, 200, 9), rotation=0, size=16)
    if LSTM==True:
        plt.yticks(np.linspace(0.05, 0.1, 5), rotation=0, size=16)
    else:
        plt.yticks(np.linspace(0.1, 0.7, 7), rotation=0, size=16)
    plt.xlabel('training steps', fontsize=16)
    plt.ylabel('loss', fontsize=16)    
    
    if not save_name==None:
        plt.savefig(os.path.join(result_pth, save_name), dpi=300)
    #plt.show()


def generate_one_hot_mt(preds, n_classes):
    res_mt = np.zeros((preds.shape[0], n_classes))
    for i, p in enumerate(preds):
        res_mt[i][p] = 1
    return res_mt


def find_index(lst, fn): # find all indexes of elements in a list according to the conditional function fn
    res = []
    for i, x in enumerate(lst):
        if fn(x):
            res.append(i)
    return res


def find_figlabel(i):
    if i==0:
        config = 'CG-NSP'
    elif i==1:
        config = 'CG-SP'
    elif i==2:
        config = 'FG-NSP'
    else:
        config = 'FG-SP' 
    return config

def evaluate_curves(ytest, scores, n_classes, auc=True):
    def compute_macro(n_classes, x_dic, y_dic): # x is fpr, x is is recall
        all_x = np.unique(np.concatenate([x_dic[i] for i in range(n_classes)]))
        mean_y = np.zeros_like(all_x)
        for i in range(n_classes):
             mean_y += interp(all_x, x_dic[i], y_dic[i])
        
        mean_y /= n_classes # Finally average it and compute AUC
        x_dic['avg'] = all_x
        y_dic['avg'] = mean_y
        return x_dic['avg'], y_dic['avg']
    
    res_x = dict()
    res_y = dict()
    for i in range(n_classes):
        current_ytest = ytest[:, i].astype(int) # nd-array
        current_score = scores[:, i]        
        if auc:
            res_x[i], res_y[i], thresholds = roc_curve(current_ytest, current_score, drop_intermediate=True)
        else:
            res_x[i], res_y[i], thresholds = precision_recall_curve(current_ytest, current_score)
    
    if auc:
        res_x['avg'], res_y['avg'] = compute_macro(n_classes, res_x, res_y)  
    else:
        res_x['avg'], res_y['avg'] = compute_macro(n_classes, res_x, res_y)
    return res_x, res_y

def compute_roc_and_prs(res_df, n_classes, figure_out_path, roc_path, tf_path, pr_path, macro=True):
    ytest = res_df['ytest'].values
    ytest = generate_one_hot_mt(ytest, n_classes)
    scores = np.array(list(res_df['scores']))
    pr_scores = np.array(list(res_df['prscores']))
    
    '''compute macro or micro curves''' # the performance can be very high
    if macro==True:
        fpr, tpr = evaluate_curves(ytest, scores, n_classes, auc=True)
        precs, recalls = evaluate_curves(ytest, scores, n_classes, auc=False)
    else:
        fpr = dict()
        tpr = dict()
        precs = dict()
        recalls = dict()
        fpr['avg'], tpr['avg'], _ = roc_curve(ytest.ravel(), scores.ravel(), drop_intermediate=True)
        precs['avg'], recalls['avg'], _ = precision_recall_curve(ytest.ravel(), scores.ravel())
        precs['avg'], recalls['avg'], _ = precision_recall_curve(ytest.ravel(), pr_scores.ravel())
    
    roc_auc = dict()
    roc_auc['avg'] = auc(fpr['avg'], tpr['avg']) # add the macro-avg value at the end of the DF
    
    tfpr_df = pd.DataFrame({'fpr':fpr["avg"], 'tpr':tpr["avg"]})
    roc_df= pd.DataFrame(roc_auc.items(), columns=['key', 'value'])
    prre_df = pd.DataFrame({'precs':precs['avg'], 'recalls':recalls['avg']}) # this is the full complex version
    
    tfpr_df.to_csv(tf_path, index=None, mode='ab')
    roc_df.to_csv(roc_path, index=None, mode='ab')
    try:
        previous_predf = pd.read_csv(pr_path, header=None) # read the simplified version
    except:
        previous_predf = pd.DataFrame() # if the first csv is empty
    prre_df.to_csv(pr_path, index=None, mode='w') # write the full version into
    prre_df = pd.read_csv(pr_path, header=None, skiprows=lambda x: x > 0 and (x-1) % 25 != 0) # simplify and read
    try:
        new_predf = previous_predf.append(prre_df)
    except:
        new_predf = prre_df
    new_predf.to_csv(pr_path, index=None, header=False, mode='w')


def process_rocdfs(df, key, val):
    dfs = []
    start = 0 # no headers
    for i, row in df.iterrows():
        if not i==0 and (key in list(row) or val in list(row)):
            temp_df = df.iloc[start+1 : i]
            temp_df.columns = [key, val]
            dfs.append(temp_df)
            start = i    
        if i==len(df)-1:
            temp_df = df.iloc[start+1 : ]
            temp_df.columns = [key, val]
            dfs.append(temp_df)                            
    return dfs


def draw_roc(tf_path, roc_path, figure_out_path):    
    # get values
    roc_df = pd.read_csv(roc_path, header=None)
    roc_dfs = process_rocdfs(roc_df, 'key', 'value')
    tfpr_df = pd.read_csv(tf_path, header=None)
    tfpf_dfs = process_rocdfs(tfpr_df, 'fpr', 'tpr')
    
    plt.figure()
    for i, current_df in enumerate(tfpf_dfs): # length 4, element sequence: coarse-nosp, coarse-sp, fine-nosp, fine-sp
        fpr = np.array(current_df['fpr']).astype(float)
        tpr = np.array(current_df['tpr']).astype(float)
        temp_auc = np.around(float(roc_dfs[i].iloc[len(roc_dfs[i])-1]['value']), 3)
        config = find_figlabel(i)
        plt.plot(fpr, tpr, label='{} AUC={})'''.format(config, str(temp_auc)), linestyle=':', linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-class ROC')
    plt.legend(loc="lower right")        
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    plt.savefig(figure_out_path, dpi=300)
    plt.show()


def draw_prcurve(pr_path, figure_out_path):
    pr_df = pd.read_csv(pr_path, header=None)
    pr_dfs = process_rocdfs(pr_df, 'precs', 'recalls')
    
    plt.figure()
    for i, current_df in enumerate(pr_dfs): # length 4, element sequence: coarse-nosp, coarse-sp, fine-nosp, fine-sp
        precs = np.array(current_df['precs']).astype(float)
        recalls = np.array(current_df['recalls']).astype(float)
        config = find_figlabel(i)
        plt.plot(recalls, precs, label=config, linestyle=':', linewidth=2)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('multi-class PR')
    plt.legend(loc='lower left')        

    plt.savefig(figure_out_path, dpi=300)
    plt.show()
    

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    
    draw_roc(os.path.join(os.getcwd(), 'results/tfr_df.csv'), 
             os.path.join(os.getcwd(), 'results/roc_df.csv'), 
             os.path.join(os.getcwd(), 'roc'))
    
    draw_prcurve(os.path.join(os.getcwd(), 'results/pr_df.csv'),  
             os.path.join(os.getcwd(), 'prg'))
        
    # plot mapping performance
    result_pth = os.path.join(os.path.join(os.getcwd(),'results'), 'record res')
    
    #bert_pth = os.path.join(result_pth, 'BERT_res.xls')
    #bert_df, bert_df_detail = process_xls_sheets(bert_pth, 'BERT') #bert_sp_df, bert_nosp_df, bert_sp_detail_df, bert_nosp_detail_df
    #
    #lstm_pth = os.path.join(result_pth, 'LSTM_res.xls')
    #lstm_df, lstm_df_detail = process_xls_sheets(lstm_pth, 'BiLSTM')
    #
    #dnn_pth = os.path.join(result_pth, 'DNN_res.xls')
    #dnn_df, dnn_df_detail = process_xls_sheets(dnn_pth, 'DNN')
    #
    # total_df = pd.concat([lstm_df, dnn_df, bert_df])
    #total_df_detail = pd.concat([lstm_df_detail, dnn_df_detail, bert_df_detail])
    #
    # total_pth = os.path.join(result_pth, 'total.png')
    #total_detail_pth = os.path.join(result_pth, 'total_detail.png')
    
    #box_plot_figure(total_df, total_pth)
    #box_plot_figure(total_df_detail, total_detail_pth, detail=True)
    
    # plot loss curves
    #loss_pth = os.path.join(result_pth, 'Loss.xls')
    #xls_file = xlrd.open_workbook(loss_pth)
    #loss_plot(xls_file, 0, save_name='lstm-g.png')
    #loss_plot(xls_file, 1, save_name='lstm-d.png')
    #loss_plot(xls_file, 2, save_name='other-g.png')
    #loss_plot(xls_file, 3, save_name='other-d.png')
    
    # tx0 = 0
    # tx1 = 25
    # ty0 = 0.055
    # ty1 = 0.10
    #
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # plt.plot(sx, sy,'purple', linewidth=2)
    #
    # axins = inset_axes(ax1, width=2, height=1.5, loc='center right')
    # axins.plot(steps, loss_df['LSTM-SP'], color=flatui[0], ls='-', linewidth=1.2)
    # axins.plot(steps, loss_df['LSTM-NSP'], color=flatui[3], ls='-', linewidth=1.2)
    # axins.axis([tx0, tx1, ty0, ty1])







