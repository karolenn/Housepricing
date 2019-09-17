#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
import pandas as pd
import torch
from scipy.interpolate import Rbf
from heatmap import heatMap

#Read data from CSV file
testData=pd.read_csv('csv/test.csv')
trainData=pd.read_csv('csv/train.csv')
#Drop Id column
testData.drop("Id", axis=1, inplace = True)
trainData.drop("Id", axis=1, inplace = True)

#####Create mapping between object datatypes and numeric values for DF#####
def mapToNumericalVal(df):
    for col_name in df.columns:
        if(df[col_name].dtype == 'object'):
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes
    return df

#Map numeric values to 0-1 for DF ####
def mapToUnitDist(df):
    for col_name in df.columns:
        max_val = df[col_name].max()
        min_val = df[col_name].min() 
        df[col_name]=df[col_name].apply(lambda x: (x-min_val)/(max_val-min_val))
    return df

#Map dataframe to [0,1]-hypercube
def MapDFtoHyperCube(df):
    df = mapToNumericalVal(df)
    df = df.fillna(0)
    df = mapToUnitDist(df)
    return df  

def PassCSVToRBF(csvFile):
    df = pd.read_csv(csvFile)
    df = mapToNumericalVal(df)
    df = df.fillna(0)
    df = mapToUnitDist(df)
  #  df.to_excel("{} to rbf.xlsx".format(csvFile))
    df = df.to_numpy()
    return df

#Convert hypercube to numpy array and transpose it so RBF can accept it
def HyperCubetoRBF(cube):
    dataArray = cube.to_numpy()
    dataArray = dataArray.transpose()
    return dataArray

#rescale price from [0,1] to original values
def RescaleSalePrice(result, originalData):
    maxHousePrice = originalData['SalePrice'].max()
    minHousePrice = originalData['SalePrice'].min()
    for j in range(len(testResult)):
        result[j]=result[j]*(maxHousePrice-minHousePrice)+minHousePrice

    return result

#write dataFrame result to csv file
def writetoCSV(result, nameOfFile):
    testResultDF = pd.DataFrame(data=result)
    testResultDF.columns = ['SalePrice']
    testResultDF.drop([0])
    testResultDF.insert(0,'Id',list(range(1461,2920,1)))
    testResultDF.to_csv(nameOfFile,index=False)


"First attempt"
#Map objects, values to [0,1] values
trainDataCube = MapDFtoHyperCube(trainData) 
testDataCube = MapDFtoHyperCube(testData) 

#Convert dataframes to numpy arrays to they can be passed to RBF
trainDataArray = HyperCubetoRBF(trainDataCube)
testDataArray = HyperCubetoRBF(testDataCube)
rbf_func = Rbf(*trainDataArray,smooth=1.5)

testResult = rbf_func(*testDataArray)
#Convert results from 0-1 values to original values
testResult = RescaleSalePrice(testResult,trainData)
#Process data and write to .csv so it is ready for submission
writetoCSV(testResult,'testResult1')

"Second attempt, remove results with low correlation <4% to sale price"
trainDataCube2nd = MapDFtoHyperCube(trainData) 
testDataCube2nd = MapDFtoHyperCube(testData) 

ResultCorr = trainDataCube2nd.corr(method='pearson')
ResultCorrSales = ResultCorr['SalePrice']
LowCorrectedCorr = ResultCorrSales[np.absolute(ResultCorrSales[:])<0.04]

trainDataCube2nd = trainDataCube2nd.drop(LowCorrectedCorr.index.values, axis = 1)
testDataCube2nd = testDataCube2nd.drop(LowCorrectedCorr.index.values, axis = 1)
trainDataArray2nd = HyperCubetoRBF(trainDataCube2nd)
testDataArray2nd = HyperCubetoRBF(testDataCube2nd)
rbf_func = Rbf(*trainDataArray2nd)

testResult2nd = rbf_func(*testDataArray2nd)
#Convert results from 0-1 values to original values
testResult2nd = RescaleSalePrice(testResult2nd,trainData)
#Process data and write to .csv so it is ready for submission
writetoCSV(testResult2nd,'testResult2nd')

"Third attempt, remove columns with >15% missing value and using same method as 2nd attempt"
#Read data from CSV file
testData3rd=pd.read_csv('csv/test.csv')
trainData3rd=pd.read_csv('csv/train.csv')
#Drop Id column
testData3rd.drop("Id", axis=1, inplace = True)
trainData3rd.drop("Id", axis=1, inplace = True)
#Drop buildings with Living Area > 4000
#trainData3rd = trainData3rd[trainData3rd.GrLivArea < 4000]
#Calculate columns with missing data in %
total = trainData3rd.isnull().sum().sort_values(ascending=False)
percent = (trainData3rd.isnull().sum()/trainData3rd.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
trainData3rd = trainData3rd.drop((missing_data[missing_data['Percent']> 0.15]).index,1)
testData3rd = testData3rd.drop((missing_data[missing_data['Percent']> 0.15]).index,1)
print(trainData3rd)

trainDataCube3rd = MapDFtoHyperCube(trainData3rd) 
testDataCube3rd = MapDFtoHyperCube(testData3rd) 

ResultCorr = trainDataCube3rd.corr(method='pearson')
ResultCorrSales = ResultCorr['SalePrice']
LowCorrectedCorr = ResultCorrSales[np.absolute(ResultCorrSales[:])<0.04]

trainDataCube3rd = trainDataCube3rd.drop(LowCorrectedCorr.index.values, axis = 1)
testDataCube3rd = testDataCube3rd.drop(LowCorrectedCorr.index.values, axis = 1)
trainDataArray3rd = HyperCubetoRBF(trainDataCube3rd)
testDataArray3rd = HyperCubetoRBF(testDataCube3rd)
rbf_func = Rbf(*trainDataArray3rd, function='cubic')

testResult3rd = rbf_func(*testDataArray3rd)
#Convert results from 0-1 values to original values
testResult3rd = RescaleSalePrice(testResult3rd,trainData3rd)
#Process data and write to .csv so it is ready for submission
writetoCSV(testResult3rd,'testResult3rdcubic')