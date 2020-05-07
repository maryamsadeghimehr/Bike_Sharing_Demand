import pandas as pd
import numpy as np
import datetime
import calendar
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

def get_data():

    df = (pd
    .read_csv('../data/train.csv', parse_dates=['datetime'])
    .reset_index(drop = True)
    )
    df['traintest'] = "TRAIN"
    df_test = (pd
    .read_csv('../data/test.csv', parse_dates=['datetime'])
    .reset_index(drop = True))
    df_test['traintest'] = "TEST"
    combined_df = pd.concat(objs = [df, df_test], axis = 0, sort= True).reset_index(drop = True)
    
    return combined_df

def get_data_origin():

    df = (pd
    .read_csv('../data/train.csv', parse_dates=['datetime'])
    .reset_index(drop = True)
    )
    df['traintest'] = "TRAIN"
    df_test = (pd
    .read_csv('../data/test.csv', parse_dates=['datetime'])
    .reset_index(drop = True))
    
    return df, df_test


def scaled_df(df, target):
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    scaled_features_df = pd.DataFrame(
        scaled_features, 
        index=df.index, 
        columns=df.columns)
    if target in df.columns:
        scaled_features_df[target] = df[target]
       
    return scaled_features_df


def any_missing_data(df):

    ans = ['no','yes']
    miss = ans[sum((df.isnull().sum() > 0)*1)]

    return miss


def parse_dates(df, col):

    df['year'] = df[col].dt.year
    df['month'] = df[col].dt.month
    df['day'] = df[col].dt.day
    df['hour'] = df[col].dt.hour
    df['minute'] = df[col].dt.minute
    df['weekday'] = df[col].dt.day_name()
    df['dayofweek'] = df[col].dt.dayofweek
    df = df.drop(col, axis = 1)

    return df
    

def update_datalist(df, df_test):
    
    data_list = [df,df_test]
    combined_df = pd.concat(data_list,axis = 0, sort = True).reset_index(drop = True)

    return data_list, combined_df

def df_xpercentile(df,col,percent):
    q = np.percentile(df[col],[percent])
    df = df[df[col] < q]
    
    return df


def drop_features (data_list, features):
    
    " input should be a list"
    tmp = []
    for data in data_list:
        data = data.drop(features, axis = 1)
        tmp.append(data)
    return tmp


def target_encoding(df, col):
    
    target = 'count'
    avg = df[[col,target]].groupby(col,as_index = False ).mean().rename(columns = {target: col + 'Cnt'})
    avg[ col + 'PerCnt'] = avg[ col + 'Cnt'].apply(lambda x: x/avg[ col + 'Cnt'].sum())
    df = (df
                  .merge(
                      avg, 
                      on = col, 
                      how = 'left')
                  .drop([col,col + 'Cnt'], axis = 1)
                 )
                 
    return df


def parse_df(combined_df,ldf , ldf_test):

    target = 'count'
    df = combined_df[0:ldf].reset_index()
    df_test = combined_df[ldf: ldf + ldf_test].reset_index()

    return df, df_test.drop(target, axis = 1)


def parse_combined_df(combined_df):
    
    target = 'count'
    df = (combined_df[combined_df['traintest'] == 'TRAIN']
          .drop('traintest', axis = 1))
    df_test = (combined_df[combined_df['traintest'] == 'TEST']
               .drop('traintest', axis = 1)
               .drop(target, axis = 1))
    

    return df, df_test

def df_with_dummies(df,col):
    
    tmp_df = pd.get_dummies(df[col],prefix = col[0])
    df = (df
          .join(tmp_df)
          .drop(col, axis = 1)
         )
    
    return df


def rmsle_cv(model,X,y,n_folds):
    
     kf = KFold(n_folds, shuffle = True, random_state=42).get_n_splits(X)
     rmse = np.sqrt(-cross_val_score(model,X,y,scoring = 'neg_mean_squared_error', cv = kf))
     return (sum(rmse/n_folds))


def get_model():

    model1 = RandomForestRegressor(n_estimators=400)
    model2 = xgb.XGBRFRegressor(max_depth=3, n_estimators=400, colsample_bytree= 0.4)
    model3 = GradientBoostingRegressor(n_estimators= 400, alpha= 0.01)
    model4 = LinearRegression()
    clfs = [model1, model2, model3, model4]
    
    return clfs


def find_model_error(model, train, test):
    
    Xtrain = train.drop('count', axis = 1)
    ytrain = train['count']
    Xtest = test.drop('count', axis = 1)
    ytest = test['count']
    model = model.fit(Xtrain,ytrain)
    pred = model.predict(Xtest)
    error = np.sqrt(mean_squared_error(np.log(ytest + 1), np.log(pred + 1)))
    return model, error

def find_model_error2(model, Xtrain, ytrain, Xtest, ytest):
    
    model = model.fit(Xtrain,ytrain)
    pred = model.predict(Xtest)
    error = np.sqrt(mean_squared_error(np.log(ytest + 1), np.log(pred + 1)))
    return model, error


def optimal_model(df):

    clf_list = [RandomForestRegressor(), xgb.XGBRFRegressor()] #GradientBoostingRegressor(), LinearRegression(), 
    #######################################################
    n_folds = 4
    X = df.drop('count', axis = 1)
    y = df['count']
    kf = KFold(n_folds,shuffle=True, random_state = 42).get_n_splits(X)
    n_estimatorslist = [100, 300,500]
    maxdepthList = [3, 5, 10]
    learning_ratelist = [0.01,0.1,1]
    colsample_bytreelist = [0.4,0.6]
    gammalist = [0]
    gridBool = [True, False]
    bestscorelist = []
    best_searchlist = []
    ###################################################
    param_GridList = [
    # [{'fit_intercept': gridBool}],  # Linear regressor
        [{'n_estimators':n_estimatorslist,
        'max_depth': maxdepthList}], # for Random Forest
        [{'n_estimators':n_estimatorslist,
        'maxdepth': maxdepthList,
        'learning_rate':learning_ratelist,
        'colsample_bytree':colsample_bytreelist,
        'gamma': gammalist}] #xgb
        ]
    #####################################################
    def my_scoring(y_true, y_pred):
        error = np.sqrt(mean_squared_error(np.log(y_true) + 1, np.log(y_pred) + 1))
        return error
    my_scoring = make_scorer(my_scoring, greater_is_better=False)
    #####################################################
    for clf, params in zip(clf_list, param_GridList):
        best_search = GridSearchCV(estimator= clf,
                                param_grid=params,
                                cv=kf,
                                scoring = my_scoring)
        best_search.fit(X,y)
        bestParams = best_search.best_params_
        bestscore = round(np.sqrt(-best_search.best_score_),5)
        bestscorelist.append(bestscore)
        best_searchlist.append(best_search)
        print('model, the best params are {}, the best score is {}'
            .format(bestParams,bestscore))

    return bestscorelist, best_search



def tuning_model(X,y,model_index):

    clf_list = [RandomForestRegressor(), 
    xgb.XGBRFRegressor(),
    GradientBoostingRegressor(),
    LinearRegression()]

    #######################################################
    n_folds = 4
    kf = (
            KFold(n_folds,
        shuffle=True, 
        random_state = 42)
        .get_n_splits(X)
        )
    n_estimatorslist = [4000] # 50,100, 350,400
    maxdepthList = [3, 5, 10]
    learning_ratelist = [0.01,0.1,1]
    colsample_bytreelist = [0.4,0.6]
    gammalist = [0]
    alphalist = [0.01, 0.1,0.9,0.99]
    gridBool = [True, False]
    bestscorelist = []
    best_searchlist = []
    ###################################################
    param_GridList = [
        [{'n_estimators':n_estimatorslist,
        'max_depth': maxdepthList}], # for Random Forest
        [{'n_estimators':n_estimatorslist,
        'maxdepth': maxdepthList,
        'learning_rate':learning_ratelist,
        'colsample_bytree':colsample_bytreelist,
        'gamma': gammalist}], #xgb
        [{'n_estimators':n_estimatorslist,
        'alpha': alphalist,
        'learning_rate':learning_ratelist}],
        [{'fit_intercept': gridBool}]  # Linear regressor
        ]

    #####################################################
  
    best_search = GridSearchCV(estimator= clf_list[model_index],
                            param_grid=param_GridList[model_index],
                            cv=kf,
                            scoring = 'neg_mean_squared_error')
    best_search.fit(X,y)
    bestParams = best_search.best_params_
    bestscore = round(np.sqrt(-best_search.best_score_),5)
    bestscorelist.append(bestscore)
    best_searchlist.append(best_search)
    print('model, the best params are {}, the best score is {}'
    .format(bestParams,bestscore))

    return bestscorelist, best_search





  