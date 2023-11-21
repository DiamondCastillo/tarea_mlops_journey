import json
import numpy as np
import pandas as pd
from datetime import datetime

class Timeit:
    def __init__(self, step_name):
        self._step_name = step_name

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f'[START] {self._step_name}')
            start_time = datetime.now()
            result = func(*args, **kwargs)
            finish_time = datetime.now()
            print(f'[FINISH] {self._step_name}')
            print(f'Duration: {finish_time - start_time}\n')
            return result
        return wrapper

# -----------------------------------------------------------------
# PREPROCESSING 
# -----------------------------------------------------------------
@Timeit('CAMBIAR TIPOS.')
def preprocess_data(df):
    
    transformations_filepath = './utils/transformations.json'
    with open(transformations_filepath) as json_file:
        transformations = json.load(json_file)
 
    cols_with_nans = list(transformations["fillmissings"].keys())
    df_nan_indices = df[cols_with_nans].isna()
    df.fillna(transformations["fillmissings"], inplace=True)

    for col_name, col_type in transformations["dscols"].items():
        df[col_name] = df[col_name].astype(col_type, copy=False)

    return df, df_nan_indices
    
# ==========================================================================
# SCORING v1.1
# ==========================================================================
@Timeit('SCORE FINAL')
def scoring_data(df):
    df1 = df.copy()
    columns_filepath = './utils/columns_for_pred.json'
    with open(columns_filepath) as json_file:
        columns = json.load(json_file)

    cols_pred = list(columns.keys())
    #df_input = df[cols_pred]

    import joblib
    import pickle
    import lightgbm as lgb

    model_filepath = './utils/modelf_weight2.sav'
    with open( model_filepath, 'rb') as file:
        model = pickle.load(file)
    
    prediccion_lgb= model.predict(df1[cols_pred], num_iteration=model.best_iteration)

    colprob = ['prob_0','prob_1','prob_2','prob_3','prob_4','prob_5']
    for j,c in enumerate(colprob):
        df1[c] = prediccion_lgb[:,j].round(6) #pd.DataFrame(prediccion_lgb, columns = colprob)

    ths = np.array([0.45472 , 0.646323, 0.169871, 0.13952 , 0.851137, 0.178732])
    #df1['clase'] = np.argmax(prediccion_lgb/ths, axis = 1)
    df1['clase'] = np.argmax(prediccion_lgb/1, axis = 1)
    #model = xgboost.Booster(model_file=model_filepath)
    #model_input = xgboost.DMatrix(df_input)
    #df['PD'] = model.predict(model_input)
    #df['PD'] = df['PD'].round(3)
    #df['score'] = (1 - df['PD']) * 1000
    #df['score'] = df['score'].round(0)

    #condition = [
    #    df.score <= 877,
    #    df.score <= 941
    #]
    #choices = [3,2]
    #df['bucket'] = np.select(condition, choices, 1)
   
    return df1

# ------------------------------------------------------------------------------
# POSTPROCESS - FINAL FORMAT
# ------------------------------------------------------------------------------
@Timeit('FORMATO FINAL')
def postprocess_data(df, args):
    
    #param_code_model = 'admbpepas'
    
    df_output = pd.DataFrame()
    df_output['codmes'] = df['period']
    #df_output['tip_doc'] = df['tip_doc']
    df_output['key_value'] = df['ID']
    #df_output['cod_cuc'] = df['cod_unico_cli']
    df_output['puntuacion'] = df['clase']
    #df_output['modelo'] = param_code_model.upper()
    df_output['modelo'] =   'modelo BBBVA'#args.submodel_name.upper()
    #df_output['fec_replica'] = df['fch_creacion']
    df_output['fec_replica'] =  '' #args.fecinfo
    df_output['segmento'] = ''
    df_output['var1'] = df['prob_0']
    df_output['var2'] = df['prob_1']
    df_output['var3'] = df['prob_2']
    df_output['var4'] = df['prob_3']
    df_output['var5'] = df['prob_4']
    df_output['var6'] = df['prob_5']

    columns_filepath = './utils/columns_map.json'
    with open(columns_filepath) as json_file:
        columns_map = json.load(json_file)

    tdt_cols = list(columns_map['map_tdt_cols'].values())
    df_output_tdt = df_output[tdt_cols]

#     cols = list(columns_map['map_woe_cols'].keys())
#     woe_cols = list(columns_map['map_woe_cols'].values())
#     df_output[woe_cols] = df[cols]

#     df_output = df_output[tdt_cols + woe_cols]

    return df_output_tdt

# ==============================================================================
# ESTABILIDAD
# ==============================================================================
def control_var_cat(df_score, variables, param_codmes='desarrollo'):
    """ Control de las variables categoricas.

    Args:
        df_score (DataFrame): dataset con el score.
        variables (list): lista de variables categoricas en 'df_score'.
        param_codmes (str): fecha de ejecucion.
    """
    df_count = []
    for var_name in variables:
        df_count_ = df_score[var_name].value_counts(dropna=False)
        df_count_ = df_count_.sort_index().reset_index()
        df_count_.columns = ['estadistico', param_codmes]
        df_count_['variable'] = var_name
        df_count.append(df_count_)
    df_count = pd.concat(df_count)
    df_count['tipo'] = 'count'
    df_count = df_count[['variable', 'estadistico', 'tipo', param_codmes]]

    df_ratio = df_count.copy()
    df_ratio['tipo'] = 'ratio'

    for var_name in variables:
        mask = df_count.variable == var_name
        total_count = df_count[mask][param_codmes].sum()
        df_ratio.loc[mask, param_codmes] = df_ratio.loc[mask, param_codmes] / total_count

    df_psi = df_count.copy()
    df_psi['tipo'] = 'psi'
    df_psi[param_codmes] = None

    return df_count.append(df_ratio).append(df_psi)

def control_var_cont(df_score, col_S, param_codmes='desarrollo'):
    """ Control de las variables continuas.
    Args:
        DF_score (DataFrame): dataset
        col_S (list): lista de variables continuas
        param_codmes (str): fecha de ejecucion
    """

    df_infer_cont = df_score[col_S]
    dfc = df_infer_cont.describe().T
    dfc['MNISS'] = df_infer_cont.isna().sum()
    dfc.columns = ['N', 'Mean', 'Std', 'Min', 'Q1', 'Q2', 'Q3', 'MAX', 'NMISS']
    dfc.N = dfc.N.astype(int)
    cf = ['N', 'NMISS', 'Mean', 'Std', 'Min', 'Q1', 'Q2', 'Q3', 'MAX' ]
    df_esta = dfc[cf].copy()
    df_estax = df_esta.unstack().reset_index()
    df_estax.columns = ['estadistico' ,'variable', param_codmes]
    df_estax['tipo'] = 'continuo'
    df_estax = df_estax.sort_values(['variable', 'estadistico'])

    return df_estax[['variable', 'estadistico', 'tipo', param_codmes]]

@Timeit('ESTABILIDAD - FINAL')
def estabilidad(df, args, df_nan_indices=None):
    """ Control de las variables.

    Args:
        df(DataFrame): dataset.

    """
    # variables discretas
    col_pred_cat = ['clase']

    # variables continuas
    col_pred_cont = ['income2',
'product_1',
'payroll_timeesp_rat',
'product_3',
'age',
'bureau_risk_category_9',
'bureau_risk_category_8',
'bureau_risk_category_7',
'bureau_risk_category_6',
'ofert_2',
'rcc_entity_3_balance_amount_sum_12',
'rcc_t3p2_days_default_max_12',
'rcc_type_prod_balance_amount_mean_sum_12',
'rcc_entity_3_days_default_mean_3',
'rcc_t5p6_balance_amount_sum_3',
'rcc_entity_4_balance_amount_sum_12',
'rcc_entity_2_balance_amount_sum_3',
'rcc_entity_5_balance_amount_sum_3',
'rcc_t3p1_balance_amount_min_12',
'rcc_t3p2_balance_amount_max_6',
'digital_sum_dig_6_12',
'digital_sum_dig_1_12',
'digital_min_dig_7_12',
'digital_mean_sum_visitas_seccion_1',
'digital_std_dig_10_12',
'digital_std_dig_8_9',
'digital_min_dig_2_12',
'digital_std_dig_9_9',
'digital_min_dig_5_9',
'digital_std_dig_1_9',
'movements_mean_cant_tipo_12',
'liabilities_mean_product_1up2_1',
'liabilities_mean_product_1up2_diff_1',
'liabilities_std_product_sum12norm_6',
'liabilities_sum_product_1up2_diff_3',
'liabilities_sum_cant_product_1sh_12',
'liabilities_mean_cant_product_1sh_1',
'liabilities_mean_product_2up2_diff_3',
'liabilities_sum_product_sum12norm_diff_12',
'liabilities_min_product_rat1Tnorm_diff_9',]
    
    codmes = '' #str(df['codmes'].iloc[1])
    
    for col_name in df_nan_indices.columns:
        df.loc[df_nan_indices[col_name], [col_name]] = np.nan

    control_cat = control_var_cat(df, col_pred_cat, param_codmes=codmes)
    mask = control_cat['variable'] == 'deciles'
    control_cat.loc[mask, 'variable'] = 'pd.cut'
    control_cont = control_var_cont(df, col_pred_cont, param_codmes=codmes)
    control = control_cont.append(control_cat)

    colsf = ['codmes', 'tipo', 'variable', 'estadistico', 'flag_segmento',
        'condicion', 'valor', 'fecinformacion', 'var1', 'var2', 'var3', 'var4',
        'var5']

    control['codmes'] = codmes
    control['flag_segmento'] = ''
    control['condicion'] = ''
    control['valor'] = control[codmes].copy()
    control['fecinformacion'] = '' #args.fecinfo
    control['var1'] = ''
    control['var2'] = ''
    control['var3'] = ''
    control['var4'] = ''
    control['var5'] = ''
    control = control[colsf]

    return control
    
    