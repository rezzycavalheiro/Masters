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

