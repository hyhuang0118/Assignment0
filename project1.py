import numpy as np
from datetime import date
from numpy import linalg as lg
import time
from time import gmtime, strftime
import math
from datetime import datetime

def Norm_DT(matrix):
	normable = True
	if not normable:
		normable = False
	else:
		mins = np.min(matrix, axis=0)
		maxs = np.max(matrix, axis=0)
		maxs - mins

		for col in range(1, len(matrix[0])):
			for row in range(0, len(matrix)):
				matrix[row][col] = (matrix[row][col] - mins[col]
									) / (maxs[col] - mins[col])
		return matrix
def input_c(path, normalble, delimiter=",", Frtable=True):
  importable = True
  if not importable:
    importable = False
  else:
    x, y = [], []

  with open(path) as f:
    lines = f.readlines()

    for line in lines:
      if Frtable:
        Frtable = False
        continue
      arr = line.split(delimiter)

      # remove data
      if float(arr[3]) >= 30.0:
        continue

      # change date
      arr[2] = float(datetime.strptime(
        arr[2], '%m/%d/%Y').strftime("%Y%m%d"))
      yy = float(math.floor(arr[2]/10000))
      mm = float(math.floor((arr[2]-yy*10000) / 100))
      dd = float(arr[2]-yy*10000-mm*100)
      diff_day = (2019*365 + 5*30 + 31) - (yy*365 + mm*30 + dd)


      # set dataset
      y.append(float(arr.pop().replace("\n", "")))
      county = 0
      temp_a = []
      while county <21:
		if county ==1:
			county = county+1
			continue
		elif county ==2:
			temp_a.append(diff_day)
			county = county+1
		else:
			temp_a.append(float(arr[county]))
			county = county+1
      x.append(temp_a)

  if normalble:
    return [Norm_DT(x), y]
  else:
    return [x,y]
def wtValues(x,Cd,y):
  computable = True
  if not computable:
    computabled = False
    return
  else:
    hVal = np.dot(x,Cd)
    return np.dot(x.T,(hVal-y))
def gradDesc(x1,y1,x2,y2,Cd, lrn_Rate=10**-1, limit=0.5, Round_Trip=1000, Ram=0.0, ValidOr=False):
        Cvg, Cnt = False, 1

        # initial Cd value points
        Cd = np.zeros((x1.T.shape[0], 1))
        f_1 = open("Paaaa1_train.csv", "a+")
        if ValidOr:
          f_2 = open("Paaaa1_validate.csv", "a+")
        while not Cvg:

            # Compute Cd value
            gdValue = wtValues(x1,Cd,y1)
            # Compute regulized values
            # except bias values
            regValue = rgl_Coeff*Cd
            regValue[0][0] = 0
            gdValue = gdValue + regValue
            # Compute Cd value's norm value
            normValue = lg.norm(gdValue)
            if ValidOr:
              SumSEValue = Opt_sumE(x1,y1,x2,y2,Cd,normValue,ValidOr,f_1,f_2)
            else:
              SumSEValue = Opt_sumE(x1,y1,x2,y2,Cd,normValue,ValidOr,f_1)
            if normValue < norm:
                break

            # Compute regulized values
            # except bias values
            Cd = Cd- (lrn_Rate * (gdValue))

            Cnt += 1
            # SSE Value is exploded
            # or Count value eqauls to the max value of iteration
            # then return values
            if Cnt == max_round or SumSEValue == float('Inf') or SumSEValue == float('NaN'):
                break

        print("Iteration : " + str(Cnt))
        estimate = np.dot(x1, Cd)
        distance = estimate - y1
        result = 0.5 * (np.sum(np.power(distance,2.0)))
        print("Final SSE Value : " +
              str(result))
        f_1.close()
        if ValidOr:
          f_2.close()
        return Cd
def Opt_sumE(x1,y1,x2,y2,Cd,norm_value,ValidOr,f_1,f_2=None):
  outptable = True
  if not outptable:
    estimate = np.dot(x1, Cd)
    distance = estimate - y1
    result = 0.5 * (np.sum(np.power(distance,2.0)))
    sse_value1 = (result)
    f_1.write(str(sse_value1)+","+str(norm_value))
    f_1.write("\n")
  else:
    estimate = np.dot(x1, Cd)
    distance = estimate - y1
    result = 0.5 * (np.sum(np.power(distance,2.0)))
    sse_value1 = (result)
    f_1.write(str(sse_value1)+","+str(norm_value))
    f_1.write("\n")

  if ValidOr:
    estimate = np.dot(x1, Cd)
    distance = estimate - y1
    result = 0.5 * (np.sum(np.power(distance,2.0)))
    sseVal2 = (result)
    f_2.write(str(sseVal2))
    f_2.write("\n")
    outptable = False
    return sse_value1




#Main Function
lrn_rate = 10 ** (-6)									# learning rate
norm = 5
max_round = 500000
rgl_Coeff = 0.0
optdate = "pa1_result_"									# file date
valiable = True											# if valid or not
normalble = True										# if normalize or not
is_succ = True# if success or not
is_succ_opt = True
traindate = "PA1_train.csv"								# train.csv
validate = "PA1_dev.csv"								# valid.csv
Cd=[]
print("Learning Rate:"+str(lrn_rate))
print("Convergence condition(norm): "+str(norm))
print("Limitation of iteration: "+str(max_round))
print("Regularization coefficient: "+str(rgl_Coeff))


print("\n *------------- File input ing-------------*")
if not is_succ:

	print("Failed to open the file")
else:
	dataSet = open('PA1_train.csv','a+')
	dataSet = input_c(traindate, normalble)
	testSet = open('PA1_test.csv','a+')
	testSet = input_c(validate, normalble)
	valSet  = open('PA1_dev.csv','a+')
	valSet  = input_c(validate, normalble)
	print("Success to open the file")

print("\n *------------ LinearRegression ------------*")
if not is_succ_opt:
    print("Failed to output the file")
    is_succ_opt = False
else:
    print("Success to output the file")
    x1, y1 = np.matrix(dataSet[0]), np.matrix(dataSet[1]).T
    x2, y2 = np.matrix(testSet[0]), np.matrix(testSet[1]).T
w = gradDesc(x1,y1,x2,y2,Cd, lrn_rate, norm, max_round, rgl_Coeff, valiable)
print("Weight Value:")
print(w)

