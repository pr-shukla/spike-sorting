import numpy as np

from utils import metrics

ground_truth = np.array([1,1,1,1,3,2,2,2,3,2,1,2,3,3,3])
prediction   = np.array([2,2,2,4,4,2,6,6,8,6,0,6,8,8,5])

print(metrics(prediction,ground_truth))