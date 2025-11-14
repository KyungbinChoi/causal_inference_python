import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression

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
    formula =  f"{treatment_col} ~ " + ' + '.join([f"C({var})" if df[var].dtype.name == 'category' else var for var in covariates])
    ps_model = smf.logit(formula, data=df).fit(disp=0)
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

    mt0 = KNeighborsRegressor(n_neighbors=1).fit(untreated[[X]], untreated[Y])

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

def propensitiy_ipw(df,treatment_col:str,outcome_col:str):
    weight_t = 1/df.query(f"{treatment_col}==1")["propensity_score"]
    weight_nt = 1/(1-df.query(f"{treatment_col}==0")["propensity_score"])
    t1 = df.query(f"{treatment_col}==1")[outcome_col] 
    t0 = df.query(f"{treatment_col}==0")[outcome_col] 

    y1 = sum(t1*weight_t)/len(df)
    y0 = sum(t0*weight_nt)/len(df)

    print("E[Y1]:", y1)
    print("E[Y0]:", y0)
    print("ATE", y1 - y0)
    ATE = y1 - y0
    return ATE

def doubly_robust(df, X, T, Y):
    # 입력된 X columns에서 데이터 추출
    X_data = df[X].copy()
    
    # object 타입 컬럼 찾기
    object_cols = X_data.select_dtypes(include=['object', 'category']).columns
    
    # object 타입 컬럼이 있는 경우 더미 변수로 변환
    if len(object_cols) > 0:
        X_data = pd.get_dummies(X_data, columns=object_cols, drop_first=True)
    
    # 기존 코드 계속...
    ps_model = LogisticRegression(penalty="none",
                                  max_iter=1000).fit(X_data, df[T])
    ps = ps_model.predict_proba(X_data)[:, 1]
    
    m0 = LinearRegression().fit(X_data[df[T]==0], df.query(f"{T}==0")[Y])
    m1 = LinearRegression().fit(X_data[df[T]==1], df.query(f"{T}==1")[Y])
    
    m0_hat = m0.predict(X_data)
    m1_hat = m1.predict(X_data)

    ATE = (
        np.mean(df[T]*(df[Y] - m1_hat)/ps + m1_hat) -
        np.mean((1-df[T])*(df[Y] - m0_hat)/(1-ps) + m0_hat)
    )
    return ATE