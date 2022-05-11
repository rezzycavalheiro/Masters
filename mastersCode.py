# ----------------- Master's code --------------------

# Imports

from scipy import stats
from scipy.stats import ks_2samp
import numpy as np
from scipy.stats import ranksums 
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
import pandas as pd
import math

filename1 = '../input/meter-swapping/meter_swapping/Dataset_0.csv'
dataset0 = pd.read_csv(filename1)

filename2 = '../input/meter-swapping/meter_swapping/Change_points_Dataset_0.csv'
cp_dataset0 = pd.read_csv(filename2)
print(cp_dataset0)

# Functions

def getRealChangePoint(dataset, consumer):
    customerA_ids = dataset['Change point (idx)'].values
    return customerA_ids[consumer]

def getConsumerData(consumerId, dataset):
    consumer_data = []
    for i, each_id in enumerate(dataset['Customer_ID']):
        if consumerId == each_id:
            consumer_data = dataset.iloc[i,1:].values
            break
    return consumer_data

def truePositiveRate(ncr, ncp):
    TPRate = ncr/ncp
    return TPRate

def falsePositiveRate(nal, ncr):
    if(nal == 0):
        return 0.0
    else:
        FPRate = (nal - ncr)/nal
        return FPRate

def gMean(TPRate, fp, tn):
    GMean = math.sqrt(TPRate * (tn/(fp + tn)))
    return GMean

def highlight_col(x):
    #copy df to new - original data are not changed
    df = x.copy()
    #set by condition
    mask1 = df['Days'] == 7
    df.loc[mask1,:] = 'background-color: #eab6fa'
    mask2 = df['Days'] == 14
    df.loc[mask2,:] = 'background-color: #94e3e2'
    mask3 = df['Days'] == 21
    df.loc[mask3,:] = 'background-color: #fac198'
    mask4 = df['Days'] == 28
    df.loc[mask4,:] = 'background-color: #f79eb7'
    return df    

# Fixed variables

customerA_ids = cp_dataset0['ID Customer A'].values.tolist()
print("Customer IDs A:", customerA_ids, '\n\n')

day = 96
consecutive_days = 7 # number of consecutive days to detect a change
ncp = 20 # ncp -> 100 consumers, 20 changes (ncp = 20)


# ********** Tests **********

# Wasserstein

def wasserstein(day, consecutive_days, ncp, week_period, threshold, dataset, cp_dataset):
    print('------------------------ Wasserstein Distance ----------------------\n\n')

    # The null hypothesis is that both groups (w0 and w1) were sampled from populations with identical distributions.
    # If the P value is small (< 0.05 or 0.01 or 0.001), conclude that the two groups were sampled from populations with different distributions
    # Parameters
    window_len = day*week_period # window lenght for the statistical tests

    results = [] # store the results of the statistical test
    count = 0 # counter to verify if the distributions are different during N consecutive days
    changes = [] # store the position (day) of detected changes by the statistical test
    true_positive = [] # store all the true positives
    all_changes = []
    real_change_point = 0

    for index, cust in enumerate(customerA_ids):
        results = [] # store the results of the statistical test
        count = 0 # counter to verify if the distributions are different during N consecutive days
        changes = []
        consumer = getConsumerData(cust, dataset)
        w0 = consumer[0:window_len] # set the data of the initial window w0
        for i in range(0, len(consumer), day):
            w1 = consumer[i:i+window_len]
            pvalue = wasserstein_distance(w0, w1)
            results.append(pvalue)
            
            # verify if the distributions are different during N consecutive days
            if pvalue > threshold: # which means w0 and w1 are from different distributions
                count = count + 1
                if count == consecutive_days:
                    starting_point = (i - consecutive_days*96) 
                    
                    changes.append(starting_point/96) # store the day in which the change was detected N consecutive days ago
                    w0 = w1 # update w0 with most recent data from w1
                    count = 0
            else:
                count = 0
        
        # print(f'Consumer ID {cust}:\n {len(changes)} changes detected at days {changes}')
        real_change_point = int(getRealChangePoint(cp_dataset, index)/96)
        for i in changes:
            if(i in range(real_change_point - 7, real_change_point + 7, 1)):
                true_positive.append(i)
                # print('Point added: ', i)
        # print('Change happens at ', real_change_point, '\n\n')
        all_changes.append(len(changes))    
        
        # plt.figure(figsize=(20,3))
        # plt.title(f'Consumer ID: {cust}')
        # plt.plot(results)
        # plt.axhline(y = threshold, color = 'gray', linestyle = 'dotted', alpha = 0.5, label='Threshold')
        # for i in range(len(changes)):
        #     plt.axvline(x=changes[i], color='g', lw=2, alpha = 0.5, linestyle = '--', label="Change points")
        # plt.axvline(real_change_point, ymin=0.05, ymax=0.95, color='purple', ls='--', lw=2, label='Ponto de mudança')
        
        
        # plt.xlabel("Days")
        # plt.ylabel("Magnitude")
        # plt.legend(['p-value','Threshold', 'Detected change'], ncol=3, loc='upper center')

    all_changes = sum(all_changes)    
    # print(true_positive)

    TPR = truePositiveRate(len(true_positive), ncp)
    FPR = falsePositiveRate(all_changes, len(true_positive))

    # print(f'True Positive Rate: {wTPR:.2f}')
    # print(f'False Positive Rate: {wFPR:.2f}')
    # print('\n\n')

    return TPR, FPR, threshold, week_period

def ks2sample(day, consecutive_days, ncp, week_period, threshold, dataset, cp_dataset):
    # print('----------------------- KS 2 Sample -----------------------\n\n')

    # The null hypothesis is that both groups (w0 and w1) were sampled from populations with identical distributions.
    # If the P value is small (< 0.05 or 0.01 or 0.001), conclude that the two groups were sampled from populations with different distributions

    window_len = day*week_period # window lenght for the statistical tests

    results = [] # store the results of the statistical test
    count = 0 # counter to verify if the distributions are different during N consecutive days
    changes = [] # store the position (day) of detected changes by the statistical test
    true_positive = [] # store all the true positives
    all_changes = []
    real_change_point = 0

    for index, cust in enumerate(customerA_ids):
        results = [] # store the results of the statistical test
        count = 0 # counter to verify if the distributions are different during N consecutive days
        changes = []
        consumer = getConsumerData(cust, dataset)
        w0 = consumer[0:window_len] # set the data of the initial window w0
        for i in range(0, len(consumer), day):
            w1 = consumer[i:i+window_len]
            pvalue = ks_2samp(w0, w1)
            results.append(pvalue.pvalue)
            
            # verify if the distributions are different during N consecutive days
            if pvalue.pvalue < threshold: # which means w0 and w1 are from different distributions
                count = count + 1
                if count == consecutive_days:
                    starting_point = (i - consecutive_days*96) 
                    
                    changes.append(starting_point/96) # store the day in which the change was detected N consecutive days ago
                    w0 = w1 # update w0 with most recent data from w1
                    count = 0
            else:
                count = 0
        
        # print(f'Consumer ID {cust}:\n {len(changes)} changes detected at days {changes}')
        real_change_point = int(getRealChangePoint(cp_dataset, index)/96)
        for i in changes:
            if(i in range(real_change_point - 7, real_change_point + 7, 1)):
                true_positive.append(i)
        #         print('Point added: ', i)
        # print('Change happens at ', real_change_point, '\n\n')
        all_changes.append(len(changes))
        
        # plt.figure(figsize=(20,3))
        # plt.title(f'Consumer ID: {cust}')
        # plt.plot(results)
        # plt.axhline(y = threshold, color = 'gray', linestyle = 'dotted', alpha = 0.5, label='Threshold')
        # for i in range(len(changes)):
        #     plt.axvline(x=changes[i], color='g', lw=2, alpha = 0.5, linestyle = '--', label="Change points")
        # plt.axvline(real_change_point, ymin=0.05, ymax=0.95, color='purple', ls='--', lw=2, label='Ponto de mudança')
        
        # plt.xlabel("Days")
        # plt.ylabel("Magnitude")
        # plt.legend(['p-value','Threshold', 'Detected change'], ncol=3, loc='upper center')
        
    all_changes = sum(all_changes)    
    # print(true_positive)

    TPR = truePositiveRate(len(true_positive), ncp)
    FPR = falsePositiveRate(all_changes, len(true_positive))

    # print(f'True Positive Rate: {TPR:.2f}')
    # print(f'False Positive Rate: {FPR:.2f}')
    # print('\n\n')


    return TPR, FPR, threshold, week_period

def ranksum(day, consecutive_days, ncp, week_period, threshold, dataset, cp_dataset):
    # Significant differences between sample populations have p-values ≤ 0.05
    # The null hypothesis is that both groups (w0 and w1) were sampled from populations with identical distributions.
    # If the P value is small (< 0.05 or 0.01 or 0.001), conclude that the two groups were sampled from populations with different distributions
    
    window_len = day*week_period # window lenght for the statistical tests

    results = [] # store the results of the statistical test
    count = 0 # counter to verify if the distributions are different during N consecutive days
    changes = [] # store the position (day) of detected changes by the statistical test
    true_positive = [] # store all the true positives
    all_changes = []
    real_change_point = 0

    for index, cust in enumerate(customerA_ids):
        results = [] # store the results of the statistical test
        count = 0 # counter to verify if the distributions are different during N consecutive days
        changes = []
        consumer = getConsumerData(cust, dataset)
        w0 = consumer[0:window_len] # set the data of the initial window w0
        for i in range(0, len(consumer), day):
            w1 = consumer[i:i+window_len]
            pvalue = ranksums(w0, w1)
            results.append(pvalue.pvalue)
            
            # verify if the distributions are different during N consecutive days
            if pvalue.pvalue <= threshold: # which means w0 and w1 are from different distributions
                count = count + 1
                if count == consecutive_days:
                    starting_point = (i - consecutive_days*96) 
                    
                    changes.append(starting_point/96) # store the day in which the change was detected N consecutive days ago
                    w0 = w1 # update w0 with most recent data from w1
                    count = 0
            else:
                count = 0
        
        # print(f'Consumer ID {cust}:\n {len(changes)} changes detected at days {changes}')
        for i in changes:
            if(i in range(real_change_point - 7, real_change_point + 7, 1)):
                true_positive.append(i)
                # print('Point added: ', i)
        real_change_point = int(getRealChangePoint(cp_dataset, index)/96)
        # print('Change happens at ', real_change_point, '\n\n')
        all_changes.append(len(changes))
        
        # plt.figure(figsize=(20,3))
        # plt.title(f'Consumer ID: {cust}')
        # plt.plot(results)
        # plt.axhline(y = threshold, color = 'gray', linestyle = 'dotted', alpha = 0.5, label='Threshold')
        # for i in range(len(changes)):
        #     plt.axvline(x=changes[i], color='g', lw=2, alpha = 0.5, linestyle = '--', label="Change points")
        # plt.axvline(real_change_point, ymin=0.05, ymax=0.95, color='purple', ls='--', lw=2, label='Ponto de mudança')
        
        # plt.xlabel("Days")
        # plt.ylabel("Magnitude")
        # plt.legend(['p-value','Threshold', 'Detected change'], ncol=3, loc='upper center')

    all_changes = sum(all_changes)    
    # print(true_positive)

    TPR = truePositiveRate(len(true_positive), ncp)
    FPR = falsePositiveRate(all_changes, len(true_positive))

    # print(f'True Positive Rate: {TPR:.2f}')
    # print(f'False Positive Rate: {FPR:.2f}')
    # print('\n\n')

    return TPR, FPR, threshold, week_period




print('Running tests...')

# wasserstein(day, consecutive_days, ncp, week_period, threshold)
TPR1, FPR1, threshold1, week_period1 = wasserstein(day, consecutive_days, ncp, 7, 0.3, dataset0, cp_dataset0)
TPR2, FPR2, threshold2, week_period1 = wasserstein(day, consecutive_days, ncp, 7, 0.4, dataset0, cp_dataset0)
TPR3, FPR3, threshold3, week_period1 = wasserstein(day, consecutive_days, ncp, 7, 0.5, dataset0, cp_dataset0)

TPR4, FPR4, threshold4, week_period2 = wasserstein(day, consecutive_days, ncp, 14, 0.3, dataset0, cp_dataset0)
TPR5, FPR5, threshold5, week_period2 = wasserstein(day, consecutive_days, ncp, 14, 0.4, dataset0, cp_dataset0)
TPR6, FPR6, threshold6, week_period2 = wasserstein(day, consecutive_days, ncp, 14, 0.5, dataset0, cp_dataset0)

TPR7, FPR7, threshold7, week_period3 = wasserstein(day, consecutive_days, ncp, 21, 0.3, dataset0, cp_dataset0)
TPR8, FPR8, threshold8, week_period3 = wasserstein(day, consecutive_days, ncp, 21, 0.4, dataset0, cp_dataset0)
TPR9, FPR9, threshold9, week_period3 = wasserstein(day, consecutive_days, ncp, 21, 0.5, dataset0, cp_dataset0)

TPR10, FPR10, threshold10, week_period4 = wasserstein(day, consecutive_days, ncp, 28, 0.3, dataset0, cp_dataset0)
TPR11, FPR11, threshold11, week_period4 = wasserstein(day, consecutive_days, ncp, 28, 0.4, dataset0, cp_dataset0)
TPR12, FPR12, threshold12, week_period4 = wasserstein(day, consecutive_days, ncp, 28, 0.5, dataset0, cp_dataset0)

print('Tests finished.')

print('******* Wasserstein Tests *******')

data = [[7, threshold1, TPR1, FPR1], 
       [7, threshold2, TPR2, FPR2], 
       [7, threshold3, TPR3, FPR3], 
       [14, threshold4, TPR4, FPR4],
       [14, threshold5, TPR5, FPR5],
       [14, threshold6, TPR6, FPR6],
       [21, threshold7, TPR7, FPR7],
       [21, threshold8, TPR8, FPR8],
       [21, threshold9, TPR9, FPR9],
       [28, threshold10, TPR10, FPR10],
       [28, threshold11, TPR11, FPR11],
       [28, threshold12, TPR12, FPR12]]
df = pd.DataFrame(data, columns = ['Days', 'Threshold', 'TPR', 'FPR'])
pd.options.display.float_format = '{:.2f}'.format
df.style.apply(highlight_col, axis=None).highlight_min(subset = ['FPR'], color = '#fcf39d', axis = 0).highlight_max(subset = ['TPR'], color = '#fcf39d', axis = 0)




print('Running tests...')

# ks2sample(day, consecutive_days, ncp, week_period, threshold)
TPR1, FPR1, threshold1, week_period1 = ks2sample(day, consecutive_days, ncp, 7, 0.3, dataset0, cp_dataset0)
TPR2, FPR2, threshold2, week_period1 = ks2sample(day, consecutive_days, ncp, 7, 0.4, dataset0, cp_dataset0)
TPR3, FPR3, threshold3, week_period1 = ks2sample(day, consecutive_days, ncp, 7, 0.5, dataset0, cp_dataset0)

TPR4, FPR4, threshold4, week_period2 = ks2sample(day, consecutive_days, ncp, 14, 0.3, dataset0, cp_dataset0)
TPR5, FPR5, threshold5, week_period2 = ks2sample(day, consecutive_days, ncp, 14, 0.4, dataset0, cp_dataset0)
TPR6, FPR6, threshold6, week_period2 = ks2sample(day, consecutive_days, ncp, 14, 0.5, dataset0, cp_dataset0)

TPR7, FPR7, threshold7, week_period3 = ks2sample(day, consecutive_days, ncp, 21, 0.3, dataset0, cp_dataset0)
TPR8, FPR8, threshold8, week_period3 = ks2sample(day, consecutive_days, ncp, 21, 0.4, dataset0, cp_dataset0)
TPR9, FPR9, threshold9, week_period3 = ks2sample(day, consecutive_days, ncp, 21, 0.5, dataset0, cp_dataset0)

TPR10, FPR10, threshold10, week_period4 = ks2sample(day, consecutive_days, ncp, 28, 0.3, dataset0, cp_dataset0)
TPR11, FPR11, threshold11, week_period4 = ks2sample(day, consecutive_days, ncp, 28, 0.4, dataset0, cp_dataset0)
TPR12, FPR12, threshold12, week_period4 = ks2sample(day, consecutive_days, ncp, 28, 0.5, dataset0, cp_dataset0)

print('Tests finished.')

print('******* KS 2 Sample Tests *******')

data = [[7, threshold1, TPR1, FPR1], 
       [7, threshold2, TPR2, FPR2], 
       [7, threshold3, TPR3, FPR3], 
       [14, threshold4, TPR4, FPR4],
       [14, threshold5, TPR5, FPR5],
       [14, threshold6, TPR6, FPR6],
       [21, threshold7, TPR7, FPR7],
       [21, threshold8, TPR8, FPR8],
       [21, threshold9, TPR9, FPR9],
       [28, threshold10, TPR10, FPR10],
       [28, threshold11, TPR11, FPR11],
       [28, threshold12, TPR12, FPR12]]
df = pd.DataFrame(data, columns = ['Days', 'Threshold', 'TPR', 'FPR'])
pd.options.display.float_format = '{:.2f}'.format
df.style.apply(highlight_col, axis=None).highlight_min(subset = ['FPR'], color = '#fcf39d', axis = 0).highlight_max(subset = ['TPR'], color = '#fcf39d', axis = 0)




print('Running tests...')

# ranksum(day, consecutive_days, ncp, week_period, threshold)
TPR1, FPR1, threshold1, week_period1 = ranksum(day, consecutive_days, ncp, 7, 0.001, dataset0, cp_dataset0)
TPR2, FPR2, threshold2, week_period1 = ranksum(day, consecutive_days, ncp, 7, 0.01, dataset0, cp_dataset0)
TPR3, FPR3, threshold3, week_period1 = ranksum(day, consecutive_days, ncp, 7, 0.05, dataset0, cp_dataset0)

TPR4, FPR4, threshold4, week_period2 = ranksum(day, consecutive_days, ncp, 14, 0.001, dataset0, cp_dataset0)
TPR5, FPR5, threshold5, week_period2 = ranksum(day, consecutive_days, ncp, 14, 0.01, dataset0, cp_dataset0)
TPR6, FPR6, threshold6, week_period2 = ranksum(day, consecutive_days, ncp, 14, 0.05, dataset0, cp_dataset0)

TPR7, FPR7, threshold7, week_period3 = ranksum(day, consecutive_days, ncp, 21, 0.001, dataset0, cp_dataset0)
TPR8, FPR8, threshold8, week_period3 = ranksum(day, consecutive_days, ncp, 21, 0.01, dataset0, cp_dataset0)
TPR9, FPR9, threshold9, week_period3 = ranksum(day, consecutive_days, ncp, 21, 0.05, dataset0, cp_dataset0)

TPR10, FPR10, threshold10, week_period4 = ranksum(day, consecutive_days, ncp, 28, 0.001, dataset0, cp_dataset0)
TPR11, FPR11, threshold11, week_period4 = ranksum(day, consecutive_days, ncp, 28, 0.01, dataset0, cp_dataset0)
TPR12, FPR12, threshold12, week_period4 = ranksum(day, consecutive_days, ncp, 28, 0.05, dataset0, cp_dataset0)

print('Tests finished.')

print('******* Ranksum Tests *******')

data = [[7, threshold1, TPR1, FPR1], 
       [7, threshold2, TPR2, FPR2], 
       [7, threshold3, TPR3, FPR3], 
       [14, threshold4, TPR4, FPR4],
       [14, threshold5, TPR5, FPR5],
       [14, threshold6, TPR6, FPR6],
       [21, threshold7, TPR7, FPR7],
       [21, threshold8, TPR8, FPR8],
       [21, threshold9, TPR9, FPR9],
       [28, threshold10, TPR10, FPR10],
       [28, threshold11, TPR11, FPR11],
       [28, threshold12, TPR12, FPR12]]
df = pd.DataFrame(data, columns = ['Days', 'Threshold', 'TPR', 'FPR'])
pd.options.display.float_format = '{:.2f}'.format
df.style.apply(highlight_col, axis=None).highlight_min(subset = ['FPR'], color = '#fcf39d', axis = 0).highlight_max(subset = ['TPR'], color = '#fcf39d', axis = 0)

