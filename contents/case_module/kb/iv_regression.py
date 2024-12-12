from linearmodels import IV2SLS

def get_iv_2SLS(df, Y, T, Z, X):
        
    # 도구변수와 공변량 리스트를 문자열로 변환
    Z_formula = ' + '.join(Z) if Z else ''  # 도구변수는 없을 수도 있음
    X_formula = ' + '.join(X) if X else ''  # 공변량도 없을 수도 있음

    # 도구변수와 공변량을 포함한 formula 생성
    formula = f"{Y} ~ 1"  # 결과변수와 상수항
    if X_formula:  # 공변량이 있는 경우 추가
        formula += f" + {X_formula}"
    formula += f" + [{T} ~ {Z_formula}]"  # 처치변수와 도구변수 추가

    iv_model = IV2SLS.from_formula(formula, df).fit(cov_type="unadjusted")

    print(iv_model.summary.tables[1])
    print(">>>> LATE : {}".format(iv_model.params))
    print(">>>> Confidential Interval \n {}".format(iv_model.conf_int(level=0.95)))

    LATE = iv_model.params
    return LATE