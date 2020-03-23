import numpy as np
import assfile as hp
import pandas as pd
import scipy.stats as stats
import random

TR_Data_File = 'pa3_train.csv'
Val_Data_File = 'pa3_val.csv'
data = hp.importCSV(TR_Data_File)
features = list(data.columns[:-1])
y = data.columns[-1]

#
#n = [1, 2, 5, 10, 25]
#m = 5
# part a loop: for i in n:
#depth = 2
#trees = []
#df = pd.DataFrame()

#for i in random_seed:
    #TR_Accuracy = []
    #Val_Accuracy = []
    #Data_len = len(data)
    #y_pred_TR_all = np.zeros((Data_len, i)) # part a
    #y_pred_Val_all = np.zeros((1625, i))      # part a
    #random.seed(1450)
    #random.seed(i)
    
    # range(i) for part a
    #for ii in range(n):
        #print (i, ii)
        #y_pred_TR = []
        #y_pred_Val = []
        #shuff_data = data.sample(Data_len, replace=True, random_state=1) # part a
        #tree = hp.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1) #part a

        # --accuracy --
        #TR_Data = hp.importCSV(TR_Data_File)
        #Val_Data = hp.importCSV(Val_Data_File)

        #_, y_pred_TR = hp.c_error(tree, TR_Data, y)
        #_, y_pred_Val = hp.c_error(tree, Val_Data, y)

        #y_pred_TR_all[:,ii] = y_pred_TR[:]
        #y_pred_Val_all[:,ii] = y_pred_Val[:]

    #y_pred_trn_agg, _ = stats.mode(y_pred_TR_all, axis=1)
    #y_pred_val_agg, _ = stats.mode(y_pred_Val_all, axis=1)

    #error_count_trn = hp.c_error(tree, TR_Data, y, mode='RF',y_pred= y_pred_trn_agg)
    #error_count_val = hp.c_error(tree, Val_Data, y, mode='RF',y_pred= y_pred_val_agg)

    #TR_Data_Len = len(TR_Data)
    #Val_Data_Len = len(Val_Data)
    #acc_trn = 1 - (error_count_trn / TR_Data_Len)
    #acc_val = 1 - (error_count_val / Val_Data_Len)
    #print('Depth {} Training Accuracy: {}' .format(i, acc_trn))
    #print('Depth {} Validation Accuracy: {}\n' .format(i, acc_val))
    #print('Depth {} Training Accuracy: {}' .format(i, acc_trn))
    #print('Depth {} Validation Accuracy: {}\n' .format(i, acc_val))
    #print([i],[acc_trn],[acc_val])

#
#P_b
#m = [1, 2, 5, 10, 25, 50]
#n = 15

#for i in random_seed:
    #TR_Accuracy = []
    #Val_Accuracy = []
    #y_pred_TR_all = np.zeros((len(data), n)) 
    #y_pred_Val_all = np.zeros((1625, n))      
    #random.seed(1450)
    #random.seed(i)
    


    #for ii in range(n):
        #print (i, ii)
        #y_pred_TR = []
        #y_pred_Val = []
        #Data_len = len(data)
        #shuff_data = data.sample(Data_len, replace=True)  # part b & c
        #tree = hp.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=i,bagging=1)

        # --accuracy --
        #TR_Data = hp.importCSV(TR_Data_File)
        #Val_Data = hp.importCSV(Val_Data_File)

        #_, y_pred_TR = hp.c_error(tree, TR_Data, y)
        #_, y_pred_Val = hp.c_error(tree, Val_Data, y)

        #y_pred_TR_all[:,ii] = y_pred_TR[:]
        #y_pred_Val_all[:,ii] = y_pred_Val[:]

    #y_pred_trn_agg, _ = stats.mode(y_pred_TR_all, axis=1)
    #y_pred_val_agg, _ = stats.mode(y_pred_Val_all, axis=1)

    #error_count_trn = hp.c_error(tree, TR_Data, y, mode='RF',y_pred= y_pred_trn_agg)
    #error_count_val = hp.c_error(tree, Val_Data, y, mode='RF',y_pred= y_pred_val_agg)

    #TR_Data_Len = len(TR_Data)
    #Val_Data_Len = len(Val_Data)
    #acc_trn = 1 - (error_count_trn / TR_Data_Len)
    #acc_val = 1 - (error_count_val / Val_Data_Len)
    #print('Depth {} Training Accuracy: {}' .format(i, acc_trn))
    #print('Depth {} Validation Accuracy: {}\n' .format(i, acc_val))
    #print('Depth {} Training Accuracy: {}' .format(i, acc_trn))
    #print('Depth {} Validation Accuracy: {}\n' .format(i, acc_val))
    #print([i],[acc_trn],[acc_val])

n = 25
m = 25
random_seed = [345,235,566,1039,550,1000,2304,2500]

depth = 2
trees = []
df = pd.DataFrame()
i = 0

for i in random_seed:
    TR_Accuracy = []
    Val_Accuracy = []
    Data_len = len(data)
    y_pred_TR_all = np.zeros((Data_len, n)) 
    y_pred_Val_all = np.zeros((1625, n))      
    random.seed(i)
    
    for ii in range(n):
        #print (i, ii)
        y_pred_TR = []
        y_pred_Val = []
        shuff_data = data.sample(Data_len, replace=True)
        #tree = hp.learn(shuff_data, features, y, depth, mode='DT', view_tree=False, m=i,bagging=1)
        tree = hp.study(shuff_data, features, y, depth, mode='DT', view_tree=False, m=m, bagging=1)

        # --accuracy --
        TR_Data = hp.importCSV(TR_Data_File)
        Val_Data = hp.importCSV(Val_Data_File)

        _, y_pred_TR = hp.c_error(tree, TR_Data, y)
        _, y_pred_Val = hp.c_error(tree, Val_Data, y)

        y_pred_TR_all[:,ii] = y_pred_TR[:]
        y_pred_Val_all[:,ii] = y_pred_Val[:]

    y_pred_trn_agg, _ = stats.mode(y_pred_TR_all, axis=1)
    y_pred_val_agg, _ = stats.mode(y_pred_Val_all, axis=1)

    error_count_trn = hp.c_error(tree, TR_Data, y, mode='RF',y_pred= y_pred_trn_agg)
    error_count_val = hp.c_error(tree, Val_Data, y, mode='RF',y_pred= y_pred_val_agg)
    
    TR_Data_Len = len(TR_Data)
    Val_Data_Len = len(Val_Data)
    acc_trn = 1 - (error_count_trn / TR_Data_Len)
    acc_val = 1 - (error_count_val / Val_Data_Len)
    print('Depth {} Training Accuracy: {}' .format(i, acc_trn))
    print('Depth {} Validation Accuracy: {}\n' .format(i, acc_val))
    print([i],[acc_trn],[acc_val])

