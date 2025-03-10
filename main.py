import pandas as pd

import RandomForest
import XGBoost
import CatBoost
import LightGBM
#import Hyperfast

def Write_and_Update(opened_file,str):
    opened_file.write(str)
    opened_file.flush()

def Benchmark(bench_rand = False, bench_xgb = False, bench_catboost = False, bench_lightgbm = False, bench_hypfast = False):
    f = open("log.txt", "a")
    data_derm = pd.read_csv('derm/derm.csv', delim_whitespace=True)
    labels_derm = pd.read_csv('derm/labels_derm.csv')["classe"]

    data_heart = pd.read_csv('heart/heart.csv', delim_whitespace=True)
    labels_heart = pd.read_csv('heart/labels_heart.csv')["output"]

    data_breast = pd.read_csv('breast_cancer/breast-cancer.csv', delim_whitespace=True)
    labels_breast = pd.read_csv('breast_cancer/labels_breast-cancer.csv')["diagnosis"]

    data_diabete = pd.read_csv('diabetes/diabetes.csv', delim_whitespace=True)
    labels_diabete = pd.read_csv('diabetes/labels_diabetes.csv')["Outcome"]

    data_covid = pd.read_csv('covid_19/Covid-19.csv', delim_whitespace=True)
    labels_covid = pd.read_csv('covid_19/labels_Covid-19.csv')["Condition"]

    for param_test_type in ["Default", "Random", "Grid"]:
        if(bench_rand == True):
            Write_and_Update(f, RandomForest.train_and_test(data_derm, labels_derm, "derm", True, param_test_type))
            Write_and_Update(f, RandomForest.train_and_test(data_heart, labels_heart, "Heart", False, param_test_type))
            Write_and_Update(f, RandomForest.train_and_test(data_breast, labels_breast, "Breast", False, param_test_type))
            Write_and_Update(f, RandomForest.train_and_test(data_diabete, labels_diabete, "Diabete", False, param_test_type))
            Write_and_Update(f, RandomForest.train_and_test(data_covid, labels_covid, "Covid-19", False, param_test_type))

        if (bench_xgb == True):
            Write_and_Update(f, XGBoost.train_and_test(data_derm, labels_derm, "derm", True, param_test_type))
            Write_and_Update(f, XGBoost.train_and_test(data_heart, labels_heart, "Heart", False, param_test_type))
            Write_and_Update(f, XGBoost.train_and_test(data_breast, labels_breast, "Breast", False, param_test_type))
            Write_and_Update(f, XGBoost.train_and_test(data_diabete, labels_diabete, "Diabete", False, param_test_type))
            Write_and_Update(f, XGBoost.train_and_test(data_covid, labels_covid, "Covid-19", False, param_test_type))

        if (bench_catboost == True):
            Write_and_Update(f, CatBoost.train_and_test(data_derm, labels_derm, "derm",True, param_test_type))
            Write_and_Update(f, CatBoost.train_and_test(data_heart, labels_heart, "Heart", False, param_test_type))
            Write_and_Update(f, CatBoost.train_and_test(data_breast, labels_breast, "Breast", False, param_test_type))
            Write_and_Update(f, CatBoost.train_and_test(data_diabete, labels_diabete, "Diabete", False, param_test_type))
            Write_and_Update(f, CatBoost.train_and_test(data_covid, labels_covid, "Covid-19", False, param_test_type))

        if (bench_lightgbm == True):
            Write_and_Update(f, LightGBM.train_and_test(data_derm, labels_derm, "derm", True, param_test_type))
            Write_and_Update(f, LightGBM.train_and_test(data_heart, labels_heart, "Heart", False, param_test_type))
            Write_and_Update(f, LightGBM.train_and_test(data_breast, labels_breast, "Breast", False, param_test_type))
            Write_and_Update(f, LightGBM.train_and_test(data_diabete, labels_diabete, "Diabete", False, param_test_type))
            Write_and_Update(f, LightGBM.train_and_test(data_covid, labels_covid, "Covid-19", False, param_test_type))

        #if (bench_hypfast == True):
            #Hyperfast.train_and_test(data_derm, labels_derm, "derm")
            #Hyperfast.train_and_test(data_heart, labels_heart, "Heart")
            #Hyperfast.train_and_test(data_breast, labels_breast, "Breast")
            #Hyperfast.train_and_test(data_diabete, labels_diabete, "Diabete")
            #Hyperfast.train_and_test(data_covid, labels_covid, "Covid-19")
        f.close()

Benchmark(True, True, False, True, False)