
# install packages   
import pandas as pd
import json
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
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler


# Initialize parser
msg = "Classification Problem"
parser = argparse.ArgumentParser(description = msg)
parser.add_argument("--run-all", help = "Run all models")
parser.add_argument("--run-model", nargs='+', help='<Required> Input data file path and model to run')

args = parser.parse_args()





# class
class Prediction:
    
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test,self.y_pred = None,None,None,None,None
        self.model = None
        self.model_name = None
    
    
    def feature_selection(self):
        
        # select features with missing values less than 60%
        req_cols, final_req_cols = [],[]
        X = self.X
        for col in self.X.columns:
            miss_rate = round(X[col].isna().sum()/len(X),2)
            if miss_rate < 0.6:
                req_cols.append(col)
        
        # select features with correlation > 0.2
        '''
        Improvements:
        1. Add visualizations
        2. Create different correlations for different type of variables (refer notebook)
        '''
        df = X[req_cols]
        df['target'] = self.y
#         g = sns.pairplot(df,hue = 'target', diag_kind= 'hist',
#                      vars=df.columns[:-1],
#                      plot_kws=dict(alpha=0.5), 
#                      diag_kws=dict(alpha=0.5))
#         plt.show()
        corr_matrix = df.corr()
        for col in req_cols:
            if abs(corr_matrix["target"][col])>0.2:
                final_req_cols.append(col)

        # update X dataframe which contain only selected features
        self.X = X[final_req_cols]
        
    
    def data_split(self, split=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,train_size=split)
        
    
    def data_normalization(self):
        cols = self.X.columns
        scaler = MinMaxScaler()
        scaler.fit(self.X)
        self.X = pd.DataFrame(scaler.transform(self.X))
        self.X.columns = cols
    
    def logistic_regression(self):
        self.model = LogisticRegression()
    
    
    def decision_tree(self):
        self.model = DecisionTreeClassifier()
    
    
    def multinomial_naive_bayes(self):
        self.model = MultinomialNB()
       
    
    def gaussian_naive_bayes(self):
        self.model = GaussianNB()
    
    
    def knn(self):
        self.model = KNeighborsClassifier()
    
    
    def rf(self,n_trees=100,criteria='gini',max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_trees, criterion=criteria, max_depth=max_depth)
    
    
    def xgb(self):
        self.model = XGBClassifier(objective="binary:logistic")
    
    
    def svm(self):
        self.model = SVC(gamma='auto')
    
    def gradient_boost(self,n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,max_depth=max_depth, random_state=random_state)
        
    
    
    def parameter_tuning(self,model='knn',scoring='accuracy',cv=5,given_params=False):
        
        # check if the parameter grid is given or have to use the the default one
        if given_params==False:
            # use predefined parameters for each model
            if model=='knn':
                self.model_name = 'knn'
                self.knn()
                params = [{'n_neighbors':[3,5,7,9], 
                           'weights':['uniform','distance'],
                           'leaf_size':[15,20,30]}] 
            if model=='logistic_regression':
                self.model_name = 'logistic_regression'
                self.logistic_regression()
                params = [{'penalty':['none','l2','elasticnet','l1'], 'C':[0.001,0.01,0.1,1,10,100,1000], 'fit_intercept':[True,False]}]
            if model== "decision_tree":
                self.model_name = 'decision_tree'
                self.decision_tree()
                params = [{'criterion':['gini','entropy'],'max_depth':[3,5,10,15,20,50]}]
            if model=='multinomial_naive_bayes':
                self.model_name = 'multinomial_naive_bayes'
                self.multinomial_naive_bayes()
                params = [{'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
            if model=="gaussian_naive_bayes":
                self.model_name = 'gaussian_naive_bayes'
                self.gaussian_naive_bayes()
                print("Message: No hyperparameter to tune for gaussian naive bayes, use predict() function to get predictions!")
                return
            if model=="rf":
                self.model_name = 'rf'
                self.rf()
                params = [{'n_estimators':[10,50,100,200],
                           'criterion':['gini','entropy'],
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'max_depth':[3,5,10,20]}]
            if model=="xgb":
                self.model_name = 'xgb'
                self.xgb()
                params=[{'max_depth': [3,6,9,12],
                        'subsample': [0.8,0.9,1.0]}]
            if model=='svm':
                self.model_name = 'svm'
                self.svm()
                params = [{'C': [1, 10], 'kernel': ('linear', 'rbf')}]
            
        # use parameter grid given
        else:
            params = given_params
            if model=='knn':
                self.model_name = 'knn'
                self.knn()
            if model=='logistic_regression':
                self.model_name = 'logistic_regression'
                self.logistic_regression()
            if model== "decision_tree":
                self.model_name = 'decision_tree'
                self.decision_tree()
            if model=='multinomial_naive_bayes':
                self.model_name = 'multinomial_naive_bayes'
                self.multinomial_naive_bayes()
            if model=="gaussian_naive_bayes":
                self.model_name = 'gaussian_naive_bayes'
                self.gaussian_naive_bayes()
                print("Message: No hyperparameter to tune for gaussian naive bayes, use predict() function to get predictions!")
                return        
            if model=="rf":
                self.model_name = 'rf'
                self.rf()
            if model=="xgb":
                self.model_name = 'xgb'
                self.xgb()
            if model=='svm':
                self.model_name = 'svm'
                self.svm()
                
        # initialise grid search
        gs = GridSearchCV(estimator=self.model,
                  param_grid = params,
                  scoring=scoring,
                  cv=cv,
                  verbose=0)
        
        
        # fit the data and get results
        try:
            gs.fit(self.X_train,self.y_train)
            print("best params: ",gs.best_params_)
            print("score: ",gs.score(self.X_train,self.y_train))
            self.model = gs
        except:
            print("Message: The parameters you entered doesn't match the input format. Please refer to the parameter_tuning function to understand the input format for parameter ranges")
            return
        

    def predict(self):
        # fit/train the model
        clf = self.model.fit(self.X_train, self.y_train)
        
        # make predictions
        self.y_pred = clf.predict(self.X_test)
    
    
    def performance(self,threshold=0.5):
        '''
        Improvements
        1. Add visualisations
        2. Read and explain the performannce matrix
        '''
        
        
#         # convert probability to binary output using given threshold (parameter)
#         y_pred_binary = (self.y_pred>threshold).astype(int)
#         print(y_pred_binary)
#         print(self.y_pred)
        
        # accuracy
        accuracy = accuracy_score(self.y_pred, self.y_test)
        print("accuracy:",accuracy)
        
        # confusion mat/rix
        cm = confusion_matrix(self.y_pred, self.y_test)
        tn, fp, fn, tp = cm.ravel()
        print("confusion matrix:\n",cm)
        
        # roc_auc
        roc_auc = roc_auc_score(self.y_pred, self.y_test)
        print("ROC AUC:",roc_auc)
        
        # pr_auc
        pr_auc = average_precision_score(self.y_pred, self.y_test)
        print("PR AUC:",pr_auc)


        performance_result = {'model':str(self.model_name),'accuracy':round(accuracy,2),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'roc auc':round(roc_auc,2),'pr auc':round(pr_auc,2)}
        path = '../output/performance_'+str(self.model_name)+'.json'
        with open(path, 'w') as fp:
            json.dump(performance_result, fp)


def run_model(file_path,model): 
        # read the data
        data = pd.read_csv(file_path)
        X = data[data.columns[:-1]]
        y = data[data.columns[-1]]
        
        # with parameter tuning

        # call class
        p = Prediction(X,y)

        # data normalization
        p.data_normalization()

        # feature engineering
        p.feature_selection()

        # split data into train and test
        p.data_split()

        # parameter tuning
        p.parameter_tuning(model=model)

        # make predictions
        p.predict()

        # get model performance
        p.performance()    



def main(data='data_file_path'):
    
    if args.run_all:
        file_path = args.run_all
        model_list = ['knn','logistic_regression','decision_tree','multinomial_naive_bayes','gaussian_naive_bayes','rf','xgb','svm']
        for model in model_list:
            run_model(file_path,model)
    
    if args.run_model:
        file_path = args.run_model[0]
        model = args.run_model[1]
        run_model(file_path,model)
        

    
if __name__ == '__main__':
    main()