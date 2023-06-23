import torch
import numpy as np
from torch.utils.data import DataLoader

import os
import warnings
import tqdm
import random

import catboost
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

import pandas as pd

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate(y, y_hat):
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))
    
def machine_model():
    X_train = pd.read_csv('./soccer_binary_data/train/X_train',index_col=0)
    Y_train = pd.read_csv("./soccer_binary_data/train/Y_train",index_col=0)   
    
    X_valid = pd.read_csv('./soccer_binary_data/valid/X_valid',index_col=0)
    Y_valid = pd.read_csv("./soccer_binary_data/valid/Y_valid",index_col=0)           

    X_test = pd.read_csv('./soccer_binary_data/test/X_test',index_col=0)
    Y_test = pd.read_csv("./soccer_binary_data/test/Y_test",index_col=0) 


    # 학습할 데이터의 인덱스를 생성
    indices = list(range(len(X_train)))
    random.shuffle(indices)

    # 섞인 인덱스를 사용하여 X_train과 Y_train을 매칭시킴
    X_train  = pd.DataFrame([X_train.loc[i] for i in tqdm.tqdm(indices,desc="X_train random shuffling")],columns=X_train.columns)
    Y_train  = pd.DataFrame([Y_train.loc[i] for i in tqdm.tqdm(indices,desc="Y_train random shuffling")],columns=Y_train.columns)
    
    #기존 연구는 score예측 모델 & concede예측 모델을 따로 학습시킴
    column_list = ['scores','concedes']
    Y_hat = pd.DataFrame()
    models = {}
    
    for col in column_list:
        #catboost외 다양한 기계학습 모델 사용
        #XGBoost, Randomforest, logistic, decision tree, SVM
        model = catboost.CatBoostClassifier()
        model.fit(X_train,Y_train[col], eval_set=[(X_valid,Y_valid[col])])
        models[col] = model
        
    #따로 학습시키고 각각 score & concede label과 AUC로 평가하기
    for col in column_list:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(X_test)]
        print(f"### Y: {col} ###")
        evaluate(Y_test[col], Y_hat[col])
        
    Y_hat.to_csv('./result/catboost_binary_result')
    
if __name__ == '__main__':
    print("soccer data analysis start\n\n")
    machine_model()
    print("soccer data analysis end\n\n")

