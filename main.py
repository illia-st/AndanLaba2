import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import scipy
from sklearn import linear_model


# a function to group a big sample
def GroupSample(sample: pd.Series):
    intervals = []
    array = sample.to_numpy()
    array = np.sort(array)
    intervals.append([array.sum()/array.size, array.size])
    return array, intervals

# read the data from the file
intervals = []
initial_data = pd.read_csv("doc/winequality-red.csv")
variables_to_analise = ["pH", "alcohol", "density"]

# group an initial data
data = []
names = []
first_group = True
counter = 0
for var in variables_to_analise:
    names.append(var)
    initial_sample = initial_data[var].copy(deep=True)
    sample, interval = GroupSample(initial_sample)
    intervals.append(interval)
    if first_group:
        data = np.empty([3, sample.size])
        first_group = False
    data[counter] = sample
    counter += 1

# process the data
# now we can go on with the correlational analysis

def GetCorCoef(X, Y, K_X, K_Y):
    matrix_XY_df = pd.DataFrame({
        'X': X,
        'Y': Y,
    })
    cut_X = pd.cut(X, bins=K_X)
    cut_Y = pd.cut(Y, bins=K_Y)

    matrix_XY_df['cut_X'] = cut_X
    matrix_XY_df['cut_Y'] = cut_Y

    CorrTable_df = pd.crosstab(
        index=matrix_XY_df['cut_X'],
        columns=matrix_XY_df['cut_Y'],
        rownames=['cut_X'],
        colnames=['cut_Y'])
    CorrTable_np = np.array(CorrTable_df)

    n_group_X = [np.sum(CorrTable_np[i]) for i in range(K_X)]

    n_group_Y = [np.sum(CorrTable_np[:, j]) for j in range(K_Y)]

    Xboun_mean = [(CorrTable_df.index[i].left + CorrTable_df.index[i].right) / 2 for i in range(K_X)]
    Xboun_mean[0] = (np.min(X) + CorrTable_df.index[0].right) / 2  # исправляем значения в крайних интервалах
    Xboun_mean[K_X - 1] = (CorrTable_df.index[K_X - 1].left + np.max(X)) / 2

    Yboun_mean = [(CorrTable_df.columns[j].left + CorrTable_df.columns[j].right) / 2 for j in range(K_Y)]
    Yboun_mean[0] = (np.min(Y) + CorrTable_df.columns[0].right) / 2  # исправляем значения в крайних интервалах
    Yboun_mean[K_Y - 1] = (CorrTable_df.columns[K_Y - 1].left + np.max(Y)) / 2

    Xmean_group = [np.sum(CorrTable_np[:, j_] * Xboun_mean) / n_group_Y[j_] for j_ in range(K_Y)]

    Ymean_group = [np.sum(CorrTable_np[i_] * Yboun_mean) / n_group_X[i_] for i_ in range(K_X)]

    Sum2_total_X = np.sum(n_group_X * (Xboun_mean - np.mean(X)) ** 2)

    Sum2_total_Y = np.sum(n_group_Y * (Yboun_mean - np.mean(Y)) ** 2)

    Sum2_between_group_X = np.sum(n_group_Y * (Xmean_group - np.mean(X)) ** 2)

    Sum2_between_group_Y = np.sum(n_group_X * (Ymean_group - np.mean(Y)) ** 2)

    corr_ratio_XY = math.sqrt(Sum2_between_group_Y / Sum2_total_Y)

    corr_ratio_YX = math.sqrt(Sum2_between_group_X / Sum2_total_X)

    return corr_ratio_XY, corr_ratio_YX

# for normal distributions
def correlation_cof(first: np.array, second: np.array):
    return scipy.stats.pearsonr(first, second), scipy.stats.pearsonr(second, first)

def getMaxLevelOfSignificance(corr_ratio_XY, n_X, K_X):
    F_corr_ratio_calc = (n_X - K_X) / (K_X - 1) * corr_ratio_XY ** 2 / (1 - corr_ratio_XY ** 2)
    m = K_X - 1
    n = n_X - K_X
    F_corr_ratio_table = scipy.stats.f.ppf(0.95, m, n)
    print(str(F_corr_ratio_calc) + " - розраховане значення")
    print(str(F_corr_ratio_table) + " - табличне значення при рівню значущості 0.95")
    # 1) use of correlation_relation and correlation_cof
analysis_result = []
for i in range(len(data)):
    # if_main_sample_normal = ifNormal(data[i])
    for j in range(i + 1, len(data)):
        if i != j:
            n_X = len(data[i])
            n_Y = len(data[j])

            group_int_number = lambda n: round(3.31 * math.log(n_X, 10) + 1) if round(3.31 * math.log(n_X, 10) + 1) >= 2 else 2
            K_X = group_int_number(n_X)
            K_Y = group_int_number(n_Y)
            if (names[i] == "pH" or names[j] == "pH") and (names[i] == "density" or names[j] == "density"):
                XY, YX = correlation_cof(first=data[i], second=data[j])
                getMaxLevelOfSignificance(XY.statistic, n_X, K_X)
                getMaxLevelOfSignificance(YX.statistic, n_Y, K_Y)
                analysis_result.append([XY.statistic, names[i] + "\\" + names[j]])
                analysis_result.append([YX.statistic, names[j] + "\\" + names[i]])
            else:
                XY, YX = GetCorCoef(data[i], data[j], K_X, K_Y)
                getMaxLevelOfSignificance(XY, n_X, K_X)
                getMaxLevelOfSignificance(YX, n_Y, K_Y)
                analysis_result.append([XY, names[i] + "\\" + names[j]])
                analysis_result.append([YX, names[j] + "\\" + names[i]])

print(sorted(analysis_result, key=lambda res: res[0], reverse=True))


# множинний аналіз
# create a regression function
# argument to predict

# get coefs for linear regression
def normal_equation(x, y):
    xtx = np.matmul(x.T.values, x.values)
    xtxi = np.matmul(np.linalg.inv(np.matmul(xtx.T,xtx)),xtx.T)
    xty = np.matmul(x.T.values, y.values)
    return np.matmul(xtxi, xty)

# compute R^2
def R_squarted(lin_reg_coefs, X, y):
    fitted = X.dot(lin_reg_coefs)  # X values on linear regression
    residuals = y - fitted  # difference y from fitted values
    difference = y - y.mean()  # difference y from its mean value
    rss = residuals.dot(residuals)  # compute sum of squares
    ess = difference.dot(difference)
    return 1 - (rss / ess)  # return R^2


def multyGetMaxLevelOfSignificance(R_value, n, m):
    F_computed = (R_value * (n - m - 1)) / ((1 - R_value) * m)
    m_ = m
    n_ = n - m - 1
    F_corr_ratio_table = scipy.stats.f.ppf(0.99, m_, n_)
    if F_corr_ratio_table < F_computed:
        print("The R value is significant")

multiples_R = []

print("============================================")

for i in range(len(names)):
    X = initial_data[[names[(i + 1) % 3], names[(i + 2) % 3]]]
    X.insert(0, "const", 1)
    y = initial_data[names[i]].apply(np.log)
    beta = normal_equation(X, y)
    R_value = R_squarted(beta, X, y)
    msg = str("Depended variable - " + str(names[i]) + ", independent variables - " + str(names[(i + 1) % 3]) + "\\" + str(names[(i + 2) % 3]))
    multiples_R.append([R_value, msg])
    # print(str(names[i]))
    # print(str(names[(i + 1) % 2]) + str(names[(i + 2) % 2]))
    # print(R_value)
    multyGetMaxLevelOfSignificance(R_value, len(initial_data[names[i]]), 2)
    print("============================================")

print(sorted(multiples_R, key=lambda res: res[0], reverse=True))