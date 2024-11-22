import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor

def get_propensity_socre_lr(df, treatment_col:str, covariates:list):
    """_summary_
    logistic regression 을 활용한 성향점수(propensity score) 산출
  
    Args:
        df (DataFrame): dataframe
        treatment_col (str): 처치 변수
        covariates (list): 공변량 X list
    Returns:
        DataFrame: dataframe
    """
    formula =  f"{treatment_col} ~ " + ' + '.join([f"C({var})" if df[var].dtype.name == 'category' else var for var in covariates + [treatment_col]])
    ps_model = smf.logit(formula, data=df.fit(disp=0))
    df = df.assign(propensity_score = ps_model.predict(df))

    return df

def propensity_score_matching(df, treatment_col:str, outcome_col:str):
    """KNN Matching 기반의 ATE 산출
    Args:
        df (DataFrame): dataframe
        treatment_col (str): 처치 변수
        outcome_col (str): 결과 변수
    Returns:
        DataFrame: dataframe
    """
    T = treatment_col
    Y = outcome_col
    X = "propensity_score"
    treated = df.query(f"{T}==1")
    untreated = df.query(f"{T}==0")

    mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[[X]],
                                                untreated[Y])

    mt1 = KNeighborsRegressor(n_neighbors=1).fit(treated[[X]], treated[Y])

    predicted = pd.concat([
        # find matches for the treated looking at the untreated knn model
        treated.assign(match=mt0.predict(treated[[X]])),
        
        # find matches for the untreated looking at the treated knn model
        untreated.assign(match=mt1.predict(untreated[[X]]))
    ])
    ATE = np.mean((predicted[Y] - predicted["match"])*predicted[T] 
        + (predicted["match"] - predicted[Y])*(1-predicted[T]))
    
    return ATE

def propensitiy_ipw():
    
    return