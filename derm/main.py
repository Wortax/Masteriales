import pandas as pd

import RandomForest
import XGBoost
import CatBoost
import LightGBM
#import Hyperfast


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

    f.write(RandomForest.train_and_test(data_derm, labels_derm, "derm", True, "Default"))
    f.close()
    f.write(RandomForest.train_and_test(data_derm, labels_derm, "derm", True, "Random"))
    f.write(RandomForest.train_and_test(data_derm, labels_derm, "derm", True, "Grid"))
    return

    if(bench_rand == True):
        RandomForest.train_and_test(data_derm, labels_derm, "derm", True)
        RandomForest.train_and_test(data_heart, labels_heart, "Heart")
        RandomForest.train_and_test(data_breast, labels_breast, "Breast")
        RandomForest.train_and_test(data_diabete, labels_diabete, "Diabete")
        RandomForest.train_and_test(data_covid, labels_covid, "Covid-19")

    if (bench_xgb == True):
        XGBoost.train_and_test(data_derm, labels_derm, "derm", True)
        XGBoost.train_and_test(data_heart, labels_heart, "Heart")
        XGBoost.train_and_test(data_breast, labels_breast, "Breast")
        XGBoost.train_and_test(data_diabete, labels_diabete, "Diabete")
        XGBoost.train_and_test(data_covid, labels_covid, "Covid-19")

    if (bench_catboost == True):
        CatBoost.train_and_test(data_derm, labels_derm, "derm",True)
        CatBoost.train_and_test(data_heart, labels_heart, "Heart")
        CatBoost.train_and_test(data_breast, labels_breast, "Breast")
        CatBoost.train_and_test(data_diabete, labels_diabete, "Diabete")
        CatBoost.train_and_test(data_covid, labels_covid, "Covid-19")

    if (bench_lightgbm == True):
        LightGBM.train_and_test(data_derm, labels_derm, "derm", True)
        LightGBM.train_and_test(data_heart, labels_heart, "Heart")
        LightGBM.train_and_test(data_breast, labels_breast, "Breast")
        LightGBM.train_and_test(data_diabete, labels_diabete, "Diabete")
        LightGBM.train_and_test(data_covid, labels_covid, "Covid-19")

    #if (bench_hypfast == True):
        #Hyperfast.train_and_test(data_derm, labels_derm, "derm")
        #Hyperfast.train_and_test(data_heart, labels_heart, "Heart")
        #Hyperfast.train_and_test(data_breast, labels_breast, "Breast")
        #Hyperfast.train_and_test(data_diabete, labels_diabete, "Diabete")
        #Hyperfast.train_and_test(data_covid, labels_covid, "Covid-19")
    f.close()

Benchmark(True, True, True, True, False)