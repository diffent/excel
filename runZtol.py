import pandas as pd
import numpy as np
import math
from scipy.stats import binom
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Lars
from sklearn.linear_model import ElasticNet

# data is most recent at top
rawdf = pd.read_csv('data.csv',sep='\t')

print(rawdf)

# dates cause confusion in automated computations so drop them 
# we can reference rawdf for dates

prices = rawdf.drop(['Date', 'Date.1'], axis = 1)

print(prices)

returnsDf = prices.pct_change(periods = -1)

print(returnsDf)

# try 150 window / 50 pts in reg

momentWindowSize = 150 # window size for stat computation S3
fwdRtnDaysCumulative = 5  # forward return days cumulative S6
nPointsInRegression = 50 # 
pushDownRegressionBackInTime = 1 # Excel S9... rolling window macro (routine) will modify this
nPointsToBacktest = 504 # Excel S15

def tailRatio(a, lower, upper):
    quantiles = np.quantile(a, [lower, upper])
    #print(quantiles)
    ratio = quantiles[1]/abs(quantiles[0])
    #print(ratio)
    return ratio

# test
# tailRatio([1,2,3,4,5,6,7,8,9,10], .2, .8)

# compute rolling moments for both assets

rollingMean = returnsDf.rolling(momentWindowSize).mean().shift(-(momentWindowSize-1))
rollingStdev = returnsDf.rolling(momentWindowSize).std().shift(-(momentWindowSize-1))
rollingSkew = returnsDf.rolling(momentWindowSize).skew().shift(-(momentWindowSize-1))
rollingKurt = returnsDf.rolling(momentWindowSize).kurt().shift(-(momentWindowSize-1))
rolling9505 = returnsDf.rolling(momentWindowSize).apply(lambda a: tailRatio(a, 0.05, 0.95)).shift(-(momentWindowSize-1))
rolling7525 = returnsDf.rolling(momentWindowSize).apply(lambda a: tailRatio(a, 0.25, 0.75)).shift(-(momentWindowSize-1))

# print(rolling7525)

NVDAclosertnPlus1 = returnsDf['NVDAClose'] + 1

print(NVDAclosertnPlus1)

rollingCumulativeReturnOver_Y = NVDAclosertnPlus1.rolling(fwdRtnDaysCumulative).apply(lambda a: math.prod(a)).shift(-(fwdRtnDaysCumulative-1))

print(rollingCumulativeReturnOver_Y)

targetCol = 'NVDAClose'
boostCol  = 'VGTClose'

# columns G to R in sheet

Xa = [rollingMean[targetCol].tolist(), 
     rollingStdev[targetCol].tolist(),
     rollingSkew[targetCol].tolist(),
     rollingKurt[targetCol].tolist(), 
     rolling9505[targetCol].tolist(),
     rolling7525[targetCol].tolist(),

     rollingMean[boostCol].tolist(),
     rollingStdev[boostCol].tolist(), 
     rollingSkew[boostCol].tolist(), 
     rollingKurt[boostCol].tolist(), 
     rolling9505[boostCol].tolist(), 
     rolling7525[boostCol].tolist()
    ]

X = np.transpose(np.array(Xa))

#(12, 5182) untransposed

print(X.shape)

nDayAheadBacktestResults = []

for i in range(nPointsToBacktest):

    pushDownRegressionBackInTime = i

    subX = X[(pushDownRegressionBackInTime+fwdRtnDaysCumulative):(pushDownRegressionBackInTime+fwdRtnDaysCumulative+nPointsInRegression), :]
    subY = rollingCumulativeReturnOver_Y[(pushDownRegressionBackInTime+0):(pushDownRegressionBackInTime+nPointsInRegression)]

    print("subY shape")
    print(subY.shape)

    # use to get same results as Excel LINEST model
    #reg = LinearRegression().fit(subX, subY)

    #reg = ElasticNet().fit(subX, subY)

    #reg = LassoLarsIC(criterion='bic').fit(subX, subY)

    # slight tuning via alpha
    # smaller alpha = keep more variables in the model
    # larger alpha = weed out weaker contributing variables
    # default of alpha = 1 just yields constant function most of the time
    # (all coefficients zero except for constant)
    reg = Lasso(alpha=0.0006).fit(subX, subY)

    R2 = reg.score(subX, subY)

    print(R2)

    #print("coeff")
    print("coeff", reg.coef_)

    nonzeroCount = np.count_nonzero(np.array(reg.coef_))
  
    #print("reg.coef_ len", len(reg.coef_), nonzeroCount)

    R2Adj = 1 - (1-R2)*(nPointsInRegression - 1)/(nPointsInRegression - (nonzeroCount+1) - 1)

    print("R2Adj", R2Adj)

    #print("intercept")
    #print(reg.intercept_)

    fit_cumulativeReturnOver_Yhat = reg.predict(subX)

    print("fit_cumulativeReturnOver_Yhat")
    print(fit_cumulativeReturnOver_Yhat)

    fit_cumulativeReturnOver_Yhat = reg.predict(subX)

    # col BM
    topOfX = X[0:i+1, :]

    trueForecastNotAlignedByTime = reg.predict(topOfX)

    print("col BM")
    print(trueForecastNotAlignedByTime)

    nDayAheadBacktestResults.append(trueForecastNotAlignedByTime[pushDownRegressionBackInTime])

#print("nDayAheadBacktestResults")
#print(nDayAheadBacktestResults)

print(type(rollingCumulativeReturnOver_Y))

rollingCumulativeReturnOver_Y_list = rollingCumulativeReturnOver_Y.to_list()

manualShiftActualBackInTime = ([0] * fwdRtnDaysCumulative) + rollingCumulativeReturnOver_Y_list

print("manualShiftActualBackInTime")

print(manualShiftActualBackInTime)

correctDir = [] # BQ
outsideZtol = []

ztol = 0.01

print("comparo")

for i in range(nPointsToBacktest):

    print(nDayAheadBacktestResults[i], manualShiftActualBackInTime[i])

    if i > (fwdRtnDaysCumulative-1) :
        if (nDayAheadBacktestResults[i] > 1 and manualShiftActualBackInTime[i] > 1) or (nDayAheadBacktestResults[i] < 1 and manualShiftActualBackInTime[i] < 1):
            correctDir.append(1)
        else:
            correctDir.append(0)

        if abs(nDayAheadBacktestResults[i] - 1) > ztol:
            outsideZtol.append(1)
        else:
            outsideZtol.append(0)

print("correctDir analysis")

print("outsideZtol = ", sum(outsideZtol))

correctDirAndOutsideZtol = np.logical_and(np.array(correctDir), np.array(outsideZtol))

print("correctDirAndOutsideZtol = ", sum(correctDirAndOutsideZtol))

ztolRatio = sum(correctDirAndOutsideZtol)/sum(outsideZtol)

print("ztolRatio", ztolRatio)

print("binomdist ztol", binom.cdf(sum(correctDirAndOutsideZtol), sum(outsideZtol), 0.5))

print(sum(correctDir), " / ", (nPointsToBacktest - fwdRtnDaysCumulative))
print("percentCorrect ", sum(correctDir) / (nPointsToBacktest - fwdRtnDaysCumulative))

print("binomdist all", binom.cdf(sum(correctDir), (nPointsToBacktest - fwdRtnDaysCumulative), 0.5))


