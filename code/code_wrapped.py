
# install packages   
import pandas as pd
import numpy as np
import seaborn as sns;
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClxassifier  
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

