import os
import argparse
from datetime import datetime
from utils.misc import install_requirements#, get_metrics, format_feature_metrics
from utils.processing import *

# ==============================================================================
# INPUTS
# ==============================================================================
DIR_REQUIREMENTS = './'
NAME_REQUIREMENTS = 'requirements.txt'
DIR_CODE_UTILS = './utils'
DIR_DATA_INPUT = './input'

# ==============================================================================
# OUTPUTS
# ==============================================================================
DIR_OUTPUT = './output/'

# Dataset
DIR_DATASET_OUTPUT = DIR_OUTPUT + 'dataset'

# Preprocess
DIR_PREPROCESSING_OUTPUT = DIR_OUTPUT + 'preprocessed'

# Inference
DIR_INFERENCE_OUTPUT = DIR_OUTPUT + 'inference'
DIR_INFERENCE_METRICS_OUTPUT = DIR_OUTPUT + 'inference_metrics'

# Postprocess
DIR_POSTPROCESSING_OUTPUT = DIR_OUTPUT + 'postprocessed'
DIR_POSTPROCESSING_OUTPUT_ATHENA = DIR_OUTPUT + 'postprocessed_athena'

# Estabilidad
DIR_ESTABILIDAD_OUTPUT = DIR_OUTPUT + 'estabilidad'
DIR_ESTABILIDAD_OUTPUT_ATHENA = DIR_OUTPUT + 'estabilidad_athena'


def print_perc_nulls(df):
    num_rows = df.shape[0]
    nulls_x_col = df.isna().sum().sort_values(ascending=False)
    nulls_x_col = nulls_x_col / num_rows * 100
    print('-' * 12, '% NULOS')
    print(nulls_x_col)


def print_summary(df, step_name):
    """Print a summary of DataFrame.

    Arguments:
    ---------
        df : DataFrame
        step_name : str
    """
    print(f'{step_name} summary:')
    print(df.info())
    print(f'DataFrame shape: {df.shape}')
    print(f'DataFrame columns: {df.columns}')
    print_perc_nulls(df)


@Timeit('LEER DATOS')
def read_parquet_data():
    #import dask.dataframe as dd  # pylint: disable=C0415
    import pandas as pd
    print('Reading data ...')
    columns_filepath = os.path.join(DIR_CODE_UTILS, 'columns_to_read.json')
    with open(columns_filepath) as json_file:
        columns = json.load(json_file)

    print('Parquet files: ', os.listdir(DIR_DATA_INPUT))
    col_names = list(columns.keys())
    print(f'{DIR_DATA_INPUT}/evaluation_period9.txt')
    df_input = pd.read_csv(f'{DIR_DATA_INPUT}/evaluation_period9.txt', sep = '|', usecols =col_names)
    #df_input = df_input.compute().reset_index(drop=True)

    # print_perc_nulls(df_input)

    return df_input


def save_df(df, local_dir, file_name, step_name):
    """Save a DataFrame to a CSV file.

    Arguments:
    ---------
        df : DataFrame
            DataFrame to be saved to a CSV file.
        local_dir : str
            Path to directory where the DataFrame will be stored.
        file_name : str
            File name.
        step_name : str
            Step being executed.
    """
    print(f'[START] GUARDAR {step_name}: {file_name}')
    start_time = datetime.now()
    filepath = os.path.join(local_dir, file_name)
    print(filepath)
    df.to_csv(f"{filepath}", sep='|', index=False)
    finish_time = datetime.now()
    print(f'[FINISH] GUARDAR {step_name}')
    print(f'Duration: {finish_time - start_time}\n')


# @Timeit('GUARDAR METRICAS')
# def save_metrics(df, args):
#     carga = '01'  # fixed
#     info = {
#         'codmes': args.partition[:6],
#         'model': args.model_name,
#         'sub_model': args.submodel_name,
#         'carga': carga,
#         'fecinfo': args.fecinfo
#     }
#     df_metrics = get_metrics(df)
#     df_metrics = format_feature_metrics(df_metrics, info)
#     name_metrics = args.inference_metrics_name
#     path_metrics = os.path.join(DIR_INFERENCE_METRICS_OUTPUT, name_metrics)
#     df_metrics.to_csv(path_metrics, index=False)


def main_inference(df_input):
    fch_creacion_col = ''# df_input.pop('fch_creacion')

    # Save dataset
    #save_df(df_input,
    #        local_dir=DIR_DATASET_OUTPUT,
    #        file_name=args.dataset_file_name,
    #        step_name='DATASET')

    df_input['fch_creacion'] = '' #fch_creacion_col

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------
    df_preprocessed, df_nan_indices = preprocess_data(df_input)
    print_summary(df_preprocessed, step_name='Preprocessed data')

    # --------------------------------------------------------------------------
    # SCORING
    # --------------------------------------------------------------------------
    df_infer = scoring_data(df_preprocessed)
    print_summary(df_infer, step_name='Scoring data')

    # --------------------------------------------------------------------------
    # POSTPROCESSING
    # --------------------------------------------------------------------------
    df_output_tdt = postprocess_data(df_infer, args)
    print_summary(df_output_tdt, step_name='Postprocessed data')

    # Save postprocessing ATHENA
#     save_df(df_output,
#             local_dir=DIR_POSTPROCESSING_OUTPUT_ATHENA,
#             file_name=args.predictions_file_name,
#             step_name='REPLICA ATHENA')

    # Save postprecessing TDT
    save_df(df_output_tdt,
            local_dir=DIR_OUTPUT, #DIR_POSTPROCESSING_OUTPUT,
            file_name=args.predictions_file_name,
            step_name='POSTPROCESSING TDT')

    # --------------------------------------------------------------------------
    # ESTABILIDAD
    # --------------------------------------------------------------------------
    df_estabilidad = estabilidad(df_infer, args, df_nan_indices)
    print_summary(df_estabilidad, step_name='Stability data')

    # Save stability data
    save_df(df_estabilidad,
            local_dir=DIR_OUTPUT , #DIR_ESTABILIDAD_OUTPUT,
            file_name='estabilidad_period_9.txt', #args.estabilidad_file_tng_name,
            step_name='ESTABILIDAD')

    # Save stability data ATHENA
#     save_df(df_estabilidad,
#             local_dir=DIR_ESTABILIDAD_OUTPUT_ATHENA,
#             file_name=args.estabilidad_file_tng_name,
#             step_name='REPLICA ATHENA')


if __name__ == '__main__':
    print("installing requirements...")
    install_requirements(DIR_REQUIREMENTS, NAME_REQUIREMENTS)

    # Reading args
    print("starting preprocessing...")
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default='model0')
    parser.add_argument('--submodel-name', type=str, default='model0')
    parser.add_argument('--fecinfo', type=str, default='00000000')
    parser.add_argument('--partition', type=str, default='00000000')

    parser.add_argument('--inference-metrics-name', type=str,
                        default='scr_inference_metrics_model00000000.csv')
    parser.add_argument('--predictions-file-name', type=str,
                        default='modelo_00000000.csv')
    parser.add_argument('--dataset-file-name', type=str,
                        default='modelo_00000000.csv')
    parser.add_argument('--estabilidad-file-tng-name', type=str, default='modelo_00000000.csv')

    args = parser.parse_args()
    print("args:", args)
    print("NAME_REQUIREMENTS:", NAME_REQUIREMENTS)
    print("DIR_REQUIREMENTS:", DIR_REQUIREMENTS)
    print("DIR_DATA_INPUT:", DIR_DATA_INPUT)

    # ==========================================================================
    # READ INPUT DATA
    # ==========================================================================
    print('[START] LEER DATOS')
    start_time = datetime.now()
    df_input = read_parquet_data()
    print('[FINISH] LEER DATOS')
    finish_time = datetime.now()
    print('Duration: {}\n'.format(finish_time - start_time))

    print('Input dataset Summary:')
    print(df_input.info())
    print("df_input shape: ", df_input.shape)
    print("df_input columns: ", df_input.columns)
    print('df_input')
    #print(df_input)

    main_inference(df_input)