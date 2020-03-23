import os
import pandas as pd
import assfile as ass
Val_File_Name = 'pa3_val.csv'
TR_File_Name = 'pa3_train.csv'
data = ass.importCSV(TR_File_Name)
chara= list(data.columns[:-1])
col = data.columns[-1]

depth = 2
tree = ass.study(data,chara,col,depth, mode='DT', view_tree=False)


TR_Data = ass.importCSV(TR_File_Name)
Val_Data = ass.importCSV(Val_File_Name)
col = TR_Data.columns[-1]
TR_Miss, _ = ass.c_error(tree, TR_Data, col)
Val_Miss, _ = ass.c_error(tree, Val_Data, col)
TR_Data_Len = len(TR_Data)
Val_Data_Len = len(Val_Data)
TR_Rate = 1 - (TR_Miss/TR_Data_Len)
Val_Rate = 1 - (Val_Miss/Val_Data_Len)
print('In the depth {} :  and the Training Accuracy Rate is : {}' .format(depth, TR_Rate))
print('In the depth {} :  and the Validation Accuracy Rate is: {}\n' .format(depth, Val_Rate))

print('Loading')
depths = [1, 2, 3, 4, 5, 6, 7, 8]
buildable = True
foreast = [None]*len(depths)
index = 0
for depth in depths:
	if not buildable:
		buildable = True
	else:
		print('Depth :')
		print(depth)
		data = ass.importCSV(TR_File_Name)
		chara= list(data.columns[:-1])
		col = data.columns[-1]
		tree = ass.study(data, chara, col, depth, mode='DT', view_tree=False)
		foreast[index] = tree
		index = index +1


TR_Datarn = ass.importCSV(TR_File_Name)
Val_Dataal = ass.importCSV(Val_File_Name)

TR_Accuracy = []
Val_Accuracy = []
index = 0
for depth in depths:
	if not buildable:
		buildable = True
	else:
		Path_To_Tree = foreast[index]
		index = index +1
		col = TR_Datarn.columns[-1]
		TR_Miss, _ = ass.c_error(Path_To_Tree, TR_Datarn, col)
		Val_Miss, _ = ass.c_error(Path_To_Tree, Val_Dataal, col)
		TR_Len = len(TR_Datarn)
		Val_Len = len(Val_Dataal)
		TR_Rate = 1 - (TR_Miss/TR_Len)
		Val_Rate = 1 - (Val_Miss/Val_Len)
		TR_Accuracy.append(TR_Rate)
		Val_Accuracy.append(Val_Rate)

print('Depth')
print(depths)
print(TR_Accuracy)
print(Val_Accuracy)
