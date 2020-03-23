import numpy as np
from datetime import date
from numpy import linalg as la
import time
from time import gmtime, strftime
import pandas as pd



def Data_Loading(File_title, Label_exist=True):
	Load_check = True
	if not Load_check:
		Load_check = False
		return File_title
	else:
		df = pd.read_csv(File_title, header=None)
		y = None
		if Label_exist:
			y = df[[0]]
			x = df.drop([0], axis=1)

			def translate(v):
				return 1.0 if v == 3 else -1.0
			y = y.applymap(translate)
			x = np.array(x)
			y = np.array(y)
			x = np.insert(x, 0, values=1.0, axis=1)
			y = y.T[0]
		else:
			x = df
			x = np.array(x)
			x = np.insert(x, 0, values=1.0, axis=1)
		return x, y

def Perceptron_in_Kernal(Training_X, Training_Y, Valid_X, Valid_Y, Poly_degree=3, iters=15):
	p_kernal = True
	if not p_kernal:
		p_kernal = False
		return Training_X, Training_Y, Valid_X, Valid_Y
	else:
		Accuracy_Training, Accuracy_Valid = [], []
		N = len(Training_X)
		alpha = np.zeros(N)

		Training_K = np.power((np.matmul(Training_X, Training_X.T) + 1), Poly_degree)
		Valid_K = np.power((np.matmul(Valid_X, Training_X.T) + 1), Poly_degree)
		for it in range(iters):
			for i in range(N):
				u = np.sign(np.dot(Training_K[i], np.multiply(alpha, Training_Y)))
				if Training_Y[i] * u <= 0:
					alpha[i] += 1
			K_infunction = Training_K
			pred = np.sign(np.matmul(K_infunction, np.multiply(alpha, Training_Y)))


			Difference_sum = 0
			for i in range(len(pred)):
				if pred[i] != Training_Y[i]:
					Difference_sum += 1
			temp =  1 - Difference_sum / len(Training_Y)

			Accuracy_Training.append(temp)
			K_infunction = Valid_K
			pred = np.sign(np.matmul(K_infunction, np.multiply(alpha, Training_Y)))

			Difference_sum = 0
			i = 0
			times = len(pred)
			while i < times:
				if pred[i] != Training_Y[i]:
					Difference_sum += 1
			temp =  1 - Difference_sum / len(Training_Y)

			Accuracy_Valid.append(temp)
			print("%d\t%.6f\t%.6f" % (it+1, Accuracy_Training[it], Accuracy_Valid[it]))
		return Accuracy_Training

class Perceptron:
    def __init__(self, data, data_valid):
        # the training data
        self.TR_Features = data.T[1:]                     # All Training Features
        self.TR_Outcome = data[0]                         # All Training Outcome Set
        self.TR_Label_outcome = self.TR_Outcome * -1 + 4  # Label '3' as 1 and Label '5' as -1
        self.TR_Numbers_Features = data.T.shape[0] - 1    # Numbers of Training of Features (785)
        self.TR_Numbers_Examples = data.shape[0]          # Number of Training of Examples
        self.Weights = []                                 # Weights
        self.Average_Weights = []                         # The average weights

        # the validation data
        self.VAL_Features = data_valid.T[1:]               # All Validation Features
        self.VAL_Outcome = data_valid[0]                   # All Validation outcome set
        self.VAL_Label_outcome = self.VAL_Outcome * -1 + 4 # Same as trainging
        self.VAL_Numbers_Examples = data_valid.shape[0]    # Number of Validation Examples

    def onlinePerceptron(self, iter_limit):
        on_p = True
        if not on_p:
            on_p = False
            return iter_limit
        else:
            self.Weights = np.zeros(self.TR_Numbers_Features)
            for iter in range(0, iter_limit):
                Error_Counting_1 = 0
                for i in range(0, self.TR_Numbers_Examples):
                    x = self.TR_Features[i]
                    x_weight = np.dot(self.Weights, x)
                    if (self.TR_Label_outcome[i] * x_weight <= 0):
                        self.Weights += self.TR_Label_outcome[i] * x
                        Error_Counting_1 += 1

				# Output result
                print(iter+1, end = " ")
                i = 0
                Error_Counting_2 = 0
                for i in range(self.VAL_Numbers_Examples):
                    x = self.VAL_Features[i]
                    x_weight = np.dot(self.Weights, x)
                    if (self.VAL_Label_outcome[i] * x_weight <= 0):
                        Error_Counting_2 += 1
                    temp = Error_Counting_2

                print("\t", (self.TR_Numbers_Examples - Error_Counting_1) /
											self.TR_Numbers_Examples, "\t",(self.VAL_Numbers_Examples - temp) /
											self.VAL_Numbers_Examples)
            return self.Weights

    def averagePerceptron(self, iter_limit):
        Check_Average = True
        if not Check_Average:
            Check_Average = False
            return iter_limit
        else:
            self.Weights = np.zeros(self.TR_Numbers_Features)
            self.Average_Weights = np.zeros(self.TR_Numbers_Features)
            c = 0   # This is for keeping the running average weight
            sum = 0   # This is for keeping the sum of cs
            for iter in range(0, iter_limit):
                Error_Counting_1 = 0
                for i in range(0, self.TR_Numbers_Examples):
                    x = self.TR_Features[i]
                    x_weight = np.dot(self.Weights, x)
                    if (self.TR_Label_outcome[i] * x_weight <= 0):
                        if (sum+c > 0):
                            self.Average_Weights = (sum*self.Average_Weights + c*self.Weights) / (sum+c)
                        sum += c
                        self.Weights += self.TR_Label_outcome[i] * x
                        c = 0
                    else :
                        c += 1
                    if (self.TR_Label_outcome[i] * np.dot(self.Average_Weights, x) <= 0):
                        Error_Counting_1 += 1

				# Output result
                print(iter+1, end = " ")
                i = 0
                Error_Counting_2 = 0
                for i in range(self.VAL_Numbers_Examples):
                    x = self.VAL_Features[i]
                    x_weight = np.dot(self.Weights, x)
                    if (self.VAL_Label_outcome[i] * x_weight <= 0):
                        Error_Counting_2 += 1
                temp = Error_Counting_2
                print("\t", (self.TR_Numbers_Examples - Error_Counting_1) /
											self.TR_Numbers_Examples, "\t",(self.VAL_Numbers_Examples - temp) /
											self.VAL_Numbers_Examples)
            if c > 0:
                self.Average_Weights = (sum*self.Average_Weights + c*self.Weights) / (sum+c)
            return self.Average_Weights

# *-------------------------------------------------------*
# *                   Main function                       *
# *-------------------------------------------------------*
Load_check = False
p_kernal = False
on_p = False
Check_Average = False
print("\nload data\n")
TR_File = 'pa2_train.csv'
VAL_File = 'pa2_valid.csv'
Test_File = 'pa2_test_no_label.csv'
iter_limit = 15
TR_Sample, TR_Label = Data_Loading("pa2_train.csv")
VAL_Sample, VAL_Label = Data_Loading("pa2_valid.csv")
Test_Sample, _ = Data_Loading("pa2_test_no_label.csv", Label_exist=False)

TR_Data = pd.read_csv(TR_File, header=None)
VAL_Data = pd.read_csv(VAL_File, header=None)


Dummy_Col_T = np.ones(TR_Data.shape[0])
Dummy_Col_V = np.ones(VAL_Data.shape[0])
TR_Data.insert(loc=785, column='785', value=Dummy_Col_T)
VAL_Data.insert(loc=785, column='785', value=Dummy_Col_V)

print("\nperception\n")

PCT_All_Data = Perceptron(TR_Data, VAL_Data)
q1_w = PCT_All_Data.onlinePerceptron(iter_limit)
#q2_w = PCT_All_Data.averagePerceptron(iter_limit)
#q3 = Perceptron_in_Kernal(TR_Sample, TR_Label, VAL_Sample, VAL_Label, 3, 15)
