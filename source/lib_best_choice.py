import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])

from datetime import date
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql import Window
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
import boto3

def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def path_exists_s3(bucket: str, path: str, is_file: bool):
    path = check_path_s3(path, is_file)
    files = list(boto3.session.Session().resource('s3').Bucket(bucket).objects.filter(Prefix=path))
    return len(files) > 0


def delete_folder_s3(bucket: str, folder_path: str):
    if path_exists_s3(bucket=bucket, path=folder_path, is_file=False):
        folder_path = check_path_s3(folder_path)
        boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()


def melt(
        df: sparkDF,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str = "variable", value_name: str = "value"
) -> sparkDF:
    _vars_and_vals = f.array(*(f.struct(f.lit(c).alias(var_name), f.col(c).alias(value_name)) for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals", f.explode(_vars_and_vals))
    cols = id_vars + [f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


def extract_1g_giorno(spark, INIZIO_PERIODO: date, FINE_PERIODO: date) -> sparkDF:

    df_1g_enelg = spark.sql(f"""
        SELECT * FROM 
            `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE
            cd_flow = 'ENEL-G' AND
            grandezza = 'A'
    """)

    df_1g_enelg = df_1g_enelg.withColumn('DATA', f.to_date(f.col('YMD'), 'yyyy/MM/dd'))
    df_1g_enelg = df_1g_enelg.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_1g_enelg = df_1g_enelg.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_1g_enelg = df_1g_enelg.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_1g_enelg = df_1g_enelg.dropDuplicates(subset=['POD', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_1g_enelg = df_1g_enelg.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_1g_enelg = df_1g_enelg.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g_enelg = melt(
        df=df_1g_enelg,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_GIORNO'
    )

    df_1g_enelg = df_1g_enelg.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g_enelg = df_1g_enelg.withColumnRenamed('ORA', 'ORA_GME')

    df_1g_enelg = df_1g_enelg.withColumn('FLUSSO_DATI', f.lit('1G'))

    return df_1g_enelg


def extract_1g_best(spark, INIZIO_PERIODO: date, FINE_PERIODO: date) -> sparkDF:

    df_1g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            grandezza = 'A'
    """)

    df_1g = df_1g.withColumn('DATA', f.to_date(f.col('YMD'), 'yyyy/MM/dd'))
    df_1g = df_1g.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_1g = df_1g.withColumn('RANK_ORIGINE',
        f.when(f.col('cd_flow') == 'XLSX', f.lit(7))
        .when(f.col('cd_flow') == 'RFO', f.lit(6))
        .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato') == 'E') & (f.col('validato') == 'S')), f.lit(5))
        .when(((f.col('cd_flow') == 'ENEL-M') | (f.col('cd_flow') == 'M15DL')), f.lit(4))
        .when(f.col('cd_flow') == 'ENEL-G', f.lit(3))
        .when(f.col('cd_flow') == 'SOS', f.lit(2))
        .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato') != 'E') & (f.col('validato') != 'S')), f.lit(1))
        .otherwise(f.lit(-1))
    )

    w_1g = Window.partitionBy('POD', 'YMD').orderBy(f.col('RANK_ORIGINE').desc(), f.col('TS').desc())
    df_1g = df_1g.withColumn('RANK', f.row_number().over(w_1g))

    df_1g = df_1g.filter(f.col('RANK') == 1)

    # TODO: eliminare drop_duplicates -> eliminare quello con meno energia
    df_1g = df_1g.dropDuplicates(subset=['POD', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_1g = df_1g.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_1g = df_1g.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g = melt(
        df=df_1g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_BEST'
    )

    df_1g = df_1g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g = df_1g.withColumnRenamed('ORA', 'ORA_GME')
    df_1g = df_1g.withColumn('FLUSSO_DATI', f.lit('1G'))

    return df_1g


def extract_2g_best(spark, INIZIO_PERIODO: date, FINE_PERIODO: date) -> sparkDF:

    df_2g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m2g_bc AS CURVA
        WHERE 
            grandezza = 'A' AND
            flg_best_choice = 'Y'
    """)

    df_2g = df_2g.withColumn('DATA', f.to_date(f.col('YMD'), 'yyyy/MM/dd'))
    df_2g = df_2g.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_2g = df_2g.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_2g = df_2g.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_2g = df_2g.dropDuplicates(subset=['POD', 'TS'])

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh * 4 + 1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh * 4 + qh_idx):03}')
        df_2g = df_2g.withColumn(f'con_e_{(hh + 1):02}', sum_col)

    df_2g = df_2g.select(
        f.col('POD'),
        f.col('DATA'),
        *[f.col(f'con_e_{(hh + 1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_2g = melt(
        df=df_2g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh + 1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_2G_BEST'
    )

    df_2g = df_2g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_2g = df_2g.withColumnRenamed('ORA', 'ORA_GME')
    df_2g = df_2g.withColumn('FLUSSO_DATI', f.lit('2G'))

    return df_2g


def build_recupero_na(spark, DATA_INIZIO: date, DATA_FINE: date, NOME_BUCKET: str) -> sparkDF:
    wnd_recupero = Window.partitionBy('POD', 'ORA_GME', 'GIORNO_SETTIMANA').orderBy(f.col('DATA').desc())

    ### CALENDARIO ###

    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .filter(f.col('DATA').between(DATA_INIZIO, DATA_FINE)) \
        .select(
        'DATA',
        'ORA_GME',
        'GIORNO_SETTIMANA'
    )
    ### MISURE 1G - BEST ###
    df_1g = extract_1g_best(spark, DATA_INIZIO, DATA_FINE)
    df_1g = df_1g.join(df_calendario, on=['DATA', 'ORA_GME'], how='left').dropna(subset=['ENERGIA_1G_RECUPERO_NA'])
    df_1g = df_1g.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1).drop('RANK')
    df_1g = df_1g.select('POD', 'ORA_GME', 'GIORNO_SETTIMANA', 'ENERGIA_1G_RECUPERO_NA')

    ### MISURE 2G - BEST ###
    df_2g = extract_2g_best(spark, DATA_INIZIO, DATA_FINE)
    df_2g = df_2g.join(df_calendario, on=['DATA', 'ORA_GME'], how='left')
    df_2g = df_2g.dropna(subset=['ENERGIA_2G_RECUPERO_NA'])
    df_2g = df_2g.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1).drop('RANK')
    df_2g = df_2g
    df_2g = df_2g.select('POD', 'ORA_GME', 'GIORNO_SETTIMANA', 'ENERGIA_2G_RECUPERO_NA')

    ### RECUPERO NA ###
    df_recupero_na = df_1g.join(df_2g, on=['POD', 'ORA_GME', 'GIORNO_SETTIMANA'], how='outer')

    output_path = f's3://{NOME_BUCKET}/datamodel/BEST_CHOICE_RECUPERO_NA'
    df_recupero_na.write.mode('overwrite').parquet(output_path)

    return output_path
