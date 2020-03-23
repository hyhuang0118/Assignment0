import os
import math
import copy
import numpy as np
import pandas as pd
import itertools
import assfile as hp
Val_File_Name = 'pa3_val.csv'
TR_File_Name = 'pa3_train.csv'
TestData_File_Name = 'pa3_test.csv'

def adaboost(data, features, y, L, depth):
    H = []
    Alpha = []

    N = len(data)
    i = 1
    data['D'] = np.ones(N)/N
    while i < L:
        h = hp.study(data, features, y, depth, mode='adaboost')
        mistake, mistake_w = calcError(h,data,features,y)
        alpha = .5*math.log((1 - mistake)/mistake)

        print('alpha: {}, mistake: {}'.format(alpha, mistake))
        Data_New = copy.deepcopy(data)

        i = 0
        for index,ex in data.iterrows():

            if (mistake_w[i]):
                Data_New['D'][index] = ex['D']*math.exp(alpha)
            else:
                Data_New['D'][index] = ex['D']*math.exp(-alpha)

            i = i +1
        data = Data_New

        Norm = data.sum(axis=0)['D']
        data['D'] = data['D']/Norm

        H.append(h)
        Alpha.append(alpha)

    return H, Alpha

def calcError(h,data,features,y):
    mistake_w = []
    mistake = 0

    for index,ex in data.iterrows():

        if (not (hp.rn_tree(h,ex) == ex[y])):
            mistake_w.append(1)
            mistake = mistake +ex['D']
            mistake_w.append(0)
        else:
            mistake_w.append(0)

    return mistake, mistake_w

data = hp.importCSV(TR_File_Name)
features= list(data.columns[:-1])
y = data.columns[-1]

Val_Data = hp.importCSV(Val_File_Name)
Val_Features= list(Val_Data.columns[:-1])
Val_y = Val_Data.columns[-1]


H_all = []
Alpha_all = []
what_all = []
TR_Wrong_All = []
Val_Wrong_All = []


depth = 1
learners = [1,2,5,10,15]
for L in learners:
    print('Adaboost -   depth={}, L={}'.format(depth,L))

    H, Alpha = adaboost(data,features,y,L,depth)

    H_all.append(H)
    Alpha_all.append(Alpha)
    what_all.append('L{}d{}'.format(L,depth))
    guessable = True
    TR_Count_Wrong = 0
    for index,ex in data.iterrows():
        isPoisonous = -1
        Wei_Guess = 0
        i = 0
        for tree in H:
            guess = hp.rn_tree(tree,ex)
            Wei_Guess += guess*Alpha[i]
            i += 1
        if Wei_Guess >= 0:
            if not guessable:
                isPoisonous = 0
                guessable = False
            else:
                isPoisonous = 1
        else:
            isPoisonous = -1
        guess = isPoisonous
        if guess != ex[y]:
            TR_Count_Wrong += 1
    TR_Wrong_All.append(TR_Count_Wrong)


    Val_Count_Wrong = 0
    for index,ex in Val_Data.iterrows():
        isPoisonous = -1
        Wei_Guess = 0
        i = 0
        for tree in H:
            guess = hp.rn_tree(tree,ex)
            Wei_Guess += guess*Alpha[i]
            i += 1
        if Wei_Guess >= 0:
            isPoisonous = 1
        else:
            isPoisonous = -1
        guess = isPoisonous
        if guess != ex[Val_y]:
            Val_Count_Wrong += 1
    Val_Wrong_All.append(Val_Count_Wrong)


depth = 2
learners = [6]
for L in learners:
    print('In adaboost and the  L={} ,  depth={}'.format(L,depth))

    H, Alpha = adaboost(data,features,y,L,depth)

    H_all.append(H)
    Alpha_all.append(Alpha)
    what_all.append('L{}d{}'.format(L,depth))

    TR_Count_Wrong = 0
    for index,ex in data.iterrows():
        isPoisonous = -1
        Wei_Guess = 0
        i = 0
        for tree in H:
            guess = hp.rn_tree(tree,ex)
            Wei_Guess += guess*Alpha[i]
            i += 1
        if Wei_Guess >= 0:
            isPoisonous = 1
        else:
            isPoisonous = -1
        guess = isPoisonous
        if guess != ex[y]:
            TR_Count_Wrong += 1
    TR_Wrong_All.append(TR_Count_Wrong)

    Val_Count_Wrong = 0
    for index,ex in Val_Data.iterrows():
        isPoisonous = -1
        Wei_Guess = 0
        i = 0
        for tree in H:
            guess = hp.rn_tree(tree,ex)
            Wei_Guess += guess*Alpha[i]
            i += 1
        if Wei_Guess >= 0:
            isPoisonous = 1
        else:
            isPoisonous = -1
        guess = isPoisonous
        if guess != ex[Val_y]:
            Val_Count_Wrong += 1
    Val_Wrong_All.append(Val_Count_Wrong)

print('\nRun Complete')
for i in range(len(what_all)):
    print('{}: Train_Wrong: {}%, Validation_Wrong: {}%'.format(what_all[i],\
        1-(TR_Wrong_All[i]/len(data)),\
            1-(Val_Wrong_All[i]/len(Val_Data))))

Adaboost_Best = Val_Wrong_All.index(min(Val_Wrong_All))

H = H_all[Adaboost_Best]
Alpha = Alpha_all[Adaboost_Best]

Test_Data = hp.importCSV(TestData_File_Name)
Test_Features= list(Test_Data.columns[:])

y_guess = []
for index,ex in Test_Data.iterrows():
    isPoisonous = -1
    Wei_Guess = 0
    i = 0
    for tree in H:
      guess = hp.rn_tree(tree,ex)
      Wei_Guess += guess*Alpha[i]
      i += 1
    if Wei_Guess >= 0:
      isPoisonous = 1
    else:
      isPoisonous = -1
    guess = isPoisonous
    y_guess.append(guess)

Output_File_Path = os.path.join(os.getcwd(),'pa3_test_predictions_p3.csv')
df = pd.DataFrame(y_guess)
df.to_csv(Output_File_Path, index=False, na_rep="None")
