import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

from pyspark.sql import SparkSession
import boto3
from datetime import date
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame as sparkDF
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd
from shared_lib import apply_map, melt, get_subfolders_s3, delete_folder_s3

map_tensioni = {
    'RES': 1.1
}

LISTA_PONTI_2023 = [
    [date(2023, 1, 1), 'FESTIVITA - CAPODANNO', 'ITALIA'],
    [date(2023, 1, 2), 'PONTE - CAPODANNO', 'ITALIA'],
    [date(2023, 1, 3), 'PONTE - EPIFANIA', 'ITALIA'],
    [date(2023, 1, 4), 'PONTE - EPIFANIA', 'ITALIA'],
    [date(2023, 1, 5), 'PONTE - EPIFANIA', 'ITALIA'],
    [date(2023, 1, 6), 'FESTIVITA - EPIFANIA', 'ITALIA'],
    [date(2023, 2, 20), 'PONTE - CARNEVALE', 'ITALIA'],
    [date(2023, 2, 21), 'FESTIVITA - CARNEVALE', 'ITALIA'],
    [date(2023, 4, 7), 'PONTE - PASQUA', 'ITALIA'],
    [date(2023, 4, 8), 'PONTE - PASQUA', 'ITALIA'],
    [date(2023, 4, 9), 'FESTIVITA - PASQUA', 'ITALIA'],
    [date(2023, 4, 10), 'FESTIVITA - PASQUETTA', 'ITALIA'],
    [date(2023, 4, 24), 'PONTE - LIBERAZIONE', 'ITALIA'],
    [date(2023, 4, 25), 'FESTIVITA - LIBERAZIONE', 'ITALIA'],
    [date(2023, 5, 1), 'FESTIVITA - FESTA DEL LAVORO', 'ITALIA'],
    [date(2023, 6, 1), 'PONTE - FESTA DELLA REPUBBLICA', 'ITALIA'],
    [date(2023, 6, 2), 'FESTIVITA - FESTA DELLA REPUBBLICA', 'ITALIA'],
    [date(2023, 6, 29), 'FESTIVITA - SAN PIETRO E PAOLO', 'RM'],
    [date(2023, 6, 30), 'PONTE - SAN PIETRO E PAOLO', 'RM'],
    [date(2023, 8, 14), 'PONTE - FERRAGOSTO', 'ITALIA'],
    [date(2023, 8, 15), 'FESTIVITA - FERRAGOSTO', 'ITALIA'],
    [date(2023, 11, 1), 'FESTIVITA - TUTTI I SANTI', 'ITALIA'],
    [date(2023, 11, 2), 'PONTE - TUTTI I SANTI', 'ITALIA'],
    [date(2023, 11, 3), 'PONTE - TUTTI I SANTI', 'ITALIA'],
    [date(2023, 12, 7), 'FESTIVITA - SANT AMBROGIO', 'MI'],
    [date(2023, 12, 8), 'FESTIVITA - IMMACOLATA', 'ITALIA'],
    [date(2023, 12, 21), 'PONTE - NATALE', 'ITALIA'],
    [date(2023, 12, 22), 'PONTE - NATALE', 'ITALIA'],
    [date(2023, 12, 25), 'FESTIVITA - NATALE', 'ITALIA'],
    [date(2023, 12, 26), 'FESTIVITA - SANTO STEFANO', 'ITALIA'],
    [date(2023, 12, 27), 'PONTE - NATALE', 'ITALIA'],
    [date(2023, 12, 28), 'PONTE - NATALE', 'ITALIA'],
    [date(2023, 12, 29), 'PONTE - NATALE', 'ITALIA'],
]

################
### FUNZIONI ###
################

def extract_1g_giorno(
    spark: SparkSession,
    inizio_periodo: date,
    fine_periodo: date,
    data_backtest: datetime
) -> sparkDF:

    df_1g_enelg = spark.sql(f"""
        SELECT * FROM 
            `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE
            cd_flow = 'ENEL-G' AND
            grandezza = 'A'
    """)

    df_1g_enelg = df_1g_enelg.filter(f.col('tms_pubblicazione') < data_backtest - relativedelta(days=1))

    df_1g_enelg = df_1g_enelg.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    df_1g_enelg = df_1g_enelg.filter(f.col('DATA').between(inizio_periodo, fine_periodo))

    df_1g_enelg = df_1g_enelg.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'DATA')))
    df_1g_enelg = df_1g_enelg.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_1g_enelg = df_1g_enelg.dropDuplicates(subset=['POD', 'DATA', 'TS'])

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

    return df_1g_enelg


def extract_1g_best(
    spark: SparkSession,
    inizio_periodo: date,
    fine_periodo: date,
    data_backtest: datetime,
    date_recupero_singole: tuple = tuple()
) -> sparkDF:

    df_1g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            grandezza = 'A'
    """)

    df_1g = df_1g.filter(f.col('tms_pubblicazione') < data_backtest - relativedelta(days=1))

    df_1g = df_1g.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    if len(date_recupero_singole) == 0:
        df_1g = df_1g.filter(f.col('DATA').between(inizio_periodo, fine_periodo))
    else:
        df_1g = df_1g.filter(
            (f.col('DATA').between(inizio_periodo, fine_periodo)) |
            (f.col('DATA').isin(list(date_recupero_singole)))
        )

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

    w_1g = Window.partitionBy('POD', 'DATA').orderBy(f.col('RANK_ORIGINE').desc(), f.col('TS').desc())
    df_1g = df_1g.withColumn('RANK', f.row_number().over(w_1g))

    df_1g = df_1g.filter(f.col('RANK') == 1)

    df_1g = df_1g.withColumn('GIORNO_SETTIMANA', f.dayofweek(f.col('DATA')))

    # TODO: eliminare drop_duplicates -> eliminare quello con meno energia
    df_1g = df_1g.dropDuplicates(subset=['POD', 'DATA', 'TS'])

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

    return df_1g


def extract_2g_best(
    spark: SparkSession,
    inizio_periodo: date,
    fine_periodo: date,
    data_backtest: datetime,
    date_recupero_singole: tuple = tuple()
) -> sparkDF:

    df_2g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m2g_bc AS CURVA
        WHERE 
            grandezza = 'A' AND
            flg_best_choice = 'Y'
    """)

    df_2g = df_2g.filter(f.col('tms_pubblicazione') < data_backtest - relativedelta(days=1))

    df_2g = df_2g.withColumn('DATA', f.to_date(f.col('ymd'), 'yyyy/MM/dd'))
    df_2g = df_2g.filter(f.col('DATA').between(inizio_periodo, fine_periodo))

    if len(date_recupero_singole) == 0:
        df_2g = df_2g.filter(f.col('DATA').between(inizio_periodo, fine_periodo))
    else:
        df_2g = df_2g.filter(
            (f.col('DATA').between(inizio_periodo, fine_periodo)) |
            (f.col('DATA').isin(list(date_recupero_singole)))
        )

    df_2g = df_2g.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'DATA')))
    df_2g = df_2g.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    # TODO: eliminare drop_duplicates
    df_2g = df_2g.dropDuplicates(subset=['POD', 'DATA', 'TS'])

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

    return df_2g


def get_ponti(
    spark: SparkSession
) -> sparkDF:

    schema_ponti = StructType([
        StructField('DATA', DateType(), False),
        StructField('FESTIVITA', StringType(), False),
        StructField('ZONA_FESTIVITA', StringType(), False),
    ])

    df_ponti = spark.createDataFrame(LISTA_PONTI_2023, schema=schema_ponti)

    if df_ponti.groupBy('DATA', 'ZONA_FESTIVITA').count().filter(f.col('count') > 1).count() > 0:
        raise Exception('Sono preesnti giorni doppi nelle festività')

    return df_ponti


def add_festivita(
    spark: SparkSession,
    df_best_choice: sparkDF
):
    df_ponti_italia = get_ponti(spark).filter(f.col('ZONA_FESTIVITA') == 'ITALIA') \
        .select('DATA', f.col('FESTIVITA').alias('FESTIVITA_ITALIA'))
    df_ponti_province = get_ponti(spark).filter(f.col('ZONA_FESTIVITA') != 'ITALIA') \
        .select('DATA', f.col('FESTIVITA').alias('FESTIVITA_PROVINCIA'), f.col('ZONA_FESTIVITA').alias('PROVINCIA_CRM'))

    if df_ponti_italia.groupBy('DATA').count().filter(f.col('count') > 1).count() > 0:
        raise Exception('ERRORE: nella tabella dei ponti globali ci sono date multiple')
    if df_ponti_province.groupBy('DATA', 'PROVINCIA_CRM').count().filter(f.col('count') > 1).count() > 0:
        raise Exception('ERRORE: nella tabella dei ponti sulle province ci sono date multiple')

    df_best_choice = df_best_choice.join(df_ponti_italia, on=['DATA'], how='left')
    df_best_choice = df_best_choice.join(df_ponti_province, on=['DATA', 'PROVINCIA_CRM'], how='left')

    df_best_choice = df_best_choice.withColumn('FESTIVITA', f.coalesce(f.col('FESTIVITA_PROVINCIA'), f.col('FESTIVITA_ITALIA'), f.lit('NO FESTIVITA')))\
        .drop('FESTIVITA_ITALIA', 'FESTIVITA_PROVINCIA')

    return df_best_choice

#############
### INPUT ###
#############

now = datetime.now().date()

COD_SIMULAZIONE = sys.argv[1]
TRACE_ANALISI = eval(sys.argv[2])
INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()
EFFETTURARE_RECUPERO = eval(sys.argv[5])
IS_PREVISIONE = eval(sys.argv[6])
DATA_BACKTEST = datetime.strptime(sys.argv[7], '%Y-%m-%d')

LOG_PATH_FILE = f'logs/BEST_CHOICE/COD_SIMULAZIONE={COD_SIMULAZIONE}/PERIODO={INIZIO_PERIODO.strftime("%Y%m%d")}_{FINE_PERIODO.strftime("%Y%m%d")}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'
NOME_TABELLA = 'BEST_CHOICE_POW'
NOME_BUCKET = 'eva-qa-s3-model'

tab_path = f'datamodel/{NOME_TABELLA}/'

log_string = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - INIZIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CARICO SESSIONE SPARK\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

##############################
### CONFIGURAZIONE CLUSTER ###
##############################

df_conf_cluster = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CONFIGURAZIONE_CLUSTER')
df_conf_cluster = df_conf_cluster.filter(f.col('SIMULAZIONE') == COD_SIMULAZIONE)
max_validita = df_conf_cluster.agg(f.max('VALIDITA').alias('MAX_VALIDITA')).collect()[0]['MAX_VALIDITA']

df_conf_cluster = df_conf_cluster.filter(f.col('VALIDITA') == max_validita)
df_conf_cluster = df_conf_cluster.pandas_api().set_index('ORDINAMENTO')
df_conf_cluster.sort_index(inplace=True)

if not df_conf_cluster.index.is_unique:
    raise AttributeError('Presente ORDINAMENTO con valori multipli')

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CONFIGURAZIONE CLUSTER\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

##################
### ANAGRAFICA ###
##################

nome_tabella_anagrafica = 'output/'

folder_anagrafica = get_subfolders_s3(bucket=NOME_BUCKET, path=nome_tabella_anagrafica)
folder_anagrafica = max([x.split('/')[-2] for x in folder_anagrafica])

df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET}/{nome_tabella_anagrafica}/{folder_anagrafica}')
df_anagrafica = df_anagrafica.filter(f.col('TRATTAMENTO') == 'O')
df_anagrafica = df_anagrafica.withColumn('TIPO_FLUSSO',
                                         f.when(f.col('TIPO_MISURATORE') == 'G', f.lit('2G')).otherwise(
                                             f.lit('1G')))
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INIZIO_VALIDITA'), 'yyyy-MM-dd'))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FINE_VALIDITA'), 'yyyy-MM-dd'))
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_INI_VALIDITA'))))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_FIN_VALIDITA'))))

if IS_PREVISIONE:
    df_anagrafica = df_anagrafica.filter(
        (f.col('DT_INI_VALIDITA') <= now) & (f.col('DT_FIN_VALIDITA') >= now)
    )
else:
    df_anagrafica = df_anagrafica.filter(
        (f.col('DT_INI_VALIDITA') <= FINE_PERIODO) & (f.col('DT_FIN_VALIDITA') >= INIZIO_PERIODO)
    )

# df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO',
#                                          f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(
#                                              DoubleType()))
# df_anagrafica = df_anagrafica.filter(
#     ~((f.col('DT_FIN_VALIDITA') < inizio_periodo) | (f.col('DT_INI_VALIDITA') > fine_periodo))
# )

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ANAGRAFICA\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

# TODO: non dovrebbe servire
df_anagrafica = df_anagrafica.dropDuplicates(subset=['pod', 'DT_INI_VALIDITA', 'DT_FIN_VALIDITA'])

#df_anagrafica.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC01')

###############
### CLUSTER ###
###############

df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.lit(None).cast(StringType()))

for k_rule in df_conf_cluster.index.to_numpy():
    row = df_conf_cluster.loc[int(k_rule)]
    lista_pod_singoli = row['LISTA_POD_SINGOLI']
    cluster_pod_singoli = row['CLUSTER_POD_SINGOLI']
    cluster_pod_singoli_nome = row['CLUSTER_POD_SINGOLI_NOME']
    regola_select = row['REGOLA_SELECT']
    regola_where = row['REGOLA_WHERE']
    if (lista_pod_singoli is not None) and (
            (cluster_pod_singoli is not None) or (cluster_pod_singoli_nome is not None) or (
            regola_select is not None) or (regola_where is not None)):
        raise AttributeError('ERRORE: presente LISTA POD SINGOLI assieme a informazioni incompatibili')
    if (cluster_pod_singoli is None) and (cluster_pod_singoli_nome is not None):
        raise AttributeError('ERRORE: presente NOME CLUSTER POD SINGOLI ma manca LISTA')
    if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is None):
        raise AttributeError('ERRORE: presente CLUSTER POD SINGOLI ma manca NOME')
    if (cluster_pod_singoli is not None) and (cluster_pod_singoli_nome is not None) and (
            (lista_pod_singoli is not None) or (regola_select is not None) or (regola_where is not None)):
        raise AttributeError('ERRORE: presente CLUSTER POD assieme a informazioni incompatibili')
    if (regola_select is not None) and ((lista_pod_singoli is not None) or (cluster_pod_singoli is not None) or (
            cluster_pod_singoli_nome is not None) or (regola_select is None)):
        raise AttributeError('ERRORE: presente REGOLA SELECT assieme a informazioni incompatibili')
    if lista_pod_singoli is not None:
        concat_syntax = f.concat_ws(
            '#',
            f.coalesce(f.col('ZONA'), f.lit('')),
            f.coalesce(f.col('ID_AREA_GESTIONALE'), f.lit('')),
            f.lit(''),
            f.lit(''),
            f.coalesce(f.col('POD'), f.lit(''))
        )
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(lista_pod_singoli),
                                                                   concat_syntax).otherwise(f.col('CLUSTER')))
    elif cluster_pod_singoli is not None:
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(cluster_pod_singoli),
                                                                   f.lit(cluster_pod_singoli_nome)).otherwise(
            f.col('CLUSTER')))
    else:
        if not pd.isna(regola_where):
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(
                f'CASE WHEN {regola_where} THEN {regola_select} ELSE CLUSTER END'))
        else:
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(regola_select))

df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.coalesce(f.col('CLUSTER'), f.lit('CLUSTER_MANCANTI')))

df_anagrafica = apply_map(
    df=df_anagrafica,
    col_name='ID_AREA_GESTIONALE',
    map_dict=map_tensioni,
    mantieni_mancanti=False,
    default_mancanti=1.0,
    output_col_name='KAPPA_PERDITE',
    output_format=DoubleType()
)

df_anagrafica = df_anagrafica.select(
    f.col('POD'),
    f.col('CLUSTER'),
    f.col('DT_INI_VALIDITA'),
    f.col('DT_FIN_VALIDITA'),
    f.col('CONSUMO_ANNUO'),
    f.col('KAPPA_PERDITE'),
    f.col('PROVINCIA_CRM')
)

# TODO: eliminare drop duplicates: non dovrebbero esistere pod doppi
df_anagrafica = df_anagrafica.dropDuplicates(subset=['POD'])

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CLUSTER\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

#df_anagrafica.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC02')

######################
### CALENDARIO-POD ###
######################

df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
    .filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO)) \
    .select(
        'UNIX_TIME',
        'TIMESTAMP',
        'DATA',
        'ORA_GME',
        'GIORNO_SETTIMANA'
)
df_best_choice = df_anagrafica.crossJoin(df_calendario)
df_best_choice = df_best_choice.filter(f.col('DATA').between(f.col('DT_INI_VALIDITA'), f.col('DT_FIN_VALIDITA')))

df_best_choice = add_festivita(spark, df_best_choice)

df_num_giorni_anno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
    .groupBy('ANNO').count().withColumnRenamed('count', 'NUM_GIORNO_ORA')
df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
df_best_choice = df_best_choice.withColumn('EAC', f.col('CONSUMO_ANNUO') / f.col('NUM_GIORNO_ORA')) \
    .drop('NUM_GIORNO_ORA', 'ANNO')

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CALENDARIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

#df_best_choice.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC03')

###########################
### AGGIUSTIAMO I CAMPI ###
###########################

# MISURE 1G - GIORNO #
df_1g_giorno = extract_1g_giorno(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)
df_best_choice = df_best_choice.join(df_1g_giorno, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_1g_giorno

# MISURE 1G - BEST #
df_1g_best = extract_1g_best(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)
df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_1g_best

# MISURE 2G - BEST #
df_2g_best = extract_2g_best(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)
df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_2g_best

df_best_choice = df_best_choice.cache()

if EFFETTURARE_RECUPERO:

    DELTA_GG_1G = 15
    DELTA_GG_2G = 5
    # TODO: pensare a come migliorare questa estrazione partendo dai valori contenuti in bc
    DATE_AGGIUNTIVE = tuple([x.DATA for x in get_ponti(spark).select('DATA').distinct().collect()])

    df_best_choice = df_best_choice.withColumn('CHIAVE_RECUPERO', f.coalesce(f.col('FESTIVITA'), f.col('GIORNO_SETTIMANA')))

    df_anagrafica = df_best_choice.select('POD', 'PROVINCIA_CRM').distinct()

    wnd_recupero = Window.partitionBy('POD', 'ORA_GME', 'CHIAVE_RECUPERO').orderBy(f.col('DATA').desc())

    df_1g_best = extract_1g_best(
        spark,
        inizio_periodo=INIZIO_PERIODO-relativedelta(days=DELTA_GG_1G),
        fine_periodo=FINE_PERIODO,
        data_backtest=DATA_BACKTEST,
        date_recupero_singole=DATE_AGGIUNTIVE
    )
    df_1g_best = df_1g_best.withColumnRenamed('ENERGIA_1G_BEST', 'ENERGIA_1G_RECUPERO_NA')

    df_1g_best = df_1g_best.join(df_anagrafica, on=['POD'], how='inner')
    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
        .filter(f.col('DATA').between(INIZIO_PERIODO-relativedelta(days=DELTA_GG_1G), FINE_PERIODO)) \
        .select('DATA', 'GIORNO_SETTIMANA').distinct()
    df_1g_best = df_1g_best.join(df_calendario, on=['DATA'], how='left')
    df_1g_best = add_festivita(spark, df_1g_best)
    df_1g_best = df_1g_best.withColumn('CHIAVE_RECUPERO', f.coalesce(f.col('FESTIVITA'), f.col('GIORNO_SETTIMANA')))
    df_1g_best = df_1g_best.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1)
    df_1g_best = df_1g_best.select('POD', 'ORA_GME', 'CHIAVE_RECUPERO', 'ENERGIA_1G_RECUPERO_NA')
    df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'ORA_GME', 'CHIAVE_RECUPERO'], how='left')

    #df_1g_best.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/RECUPERO_1G/')

    del df_1g_best, df_calendario

    df_2g_best = extract_2g_best(
        spark,
        inizio_periodo=INIZIO_PERIODO-relativedelta(days=DELTA_GG_2G),
        fine_periodo=FINE_PERIODO,
        data_backtest=DATA_BACKTEST,
        date_recupero_singole=DATE_AGGIUNTIVE
    )
    df_2g_best = df_2g_best.withColumnRenamed('ENERGIA_2G_BEST', 'ENERGIA_2G_RECUPERO_NA')
    df_2g_best = df_2g_best.join(df_anagrafica, on=['POD'], how='inner')
    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
        .filter(f.col('DATA').between(INIZIO_PERIODO-relativedelta(days=DELTA_GG_2G), FINE_PERIODO)) \
        .select('DATA', 'GIORNO_SETTIMANA').distinct()
    df_2g_best = df_2g_best.join(df_calendario, on=['DATA'], how='left')
    df_2g_best = add_festivita(spark, df_2g_best)
    df_2g_best = df_2g_best.withColumn('CHIAVE_RECUPERO', f.coalesce(f.col('FESTIVITA'), f.col('GIORNO_SETTIMANA')))
    df_2g_best = df_2g_best.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1)
    df_2g_best = df_2g_best.select('POD', 'ORA_GME', 'CHIAVE_RECUPERO', 'ENERGIA_2G_RECUPERO_NA')
    df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'ORA_GME', 'CHIAVE_RECUPERO'], how='left')

    #df_2g_best.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/RECUPERO_2G/')

    del df_2g_best, df_calendario

else:

    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_RECUPERO_NA', f.lit(None).cast(DoubleType()))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_RECUPERO_NA', f.lit(None).cast(DoubleType()))

###########
### POW ###
###########

df_pow = spark.sql("SELECT * FROM `623333656140/thr_prod_glue_db`.ee_programmi_vert")
w_pow = Window.partitionBy('pod', 'universal_time').orderBy(f.col('ts').desc())
df_pow = df_pow.withColumn('RANK', f.row_number().over(w_pow))
df_pow = df_pow.filter(f.col('RANK') == 1)
df_pow = df_pow.select(
    (f.col('universal_time').cast('bigint') + f.lit(63064800)).alias('UNIX_TIME'),
    f.col('pod').alias('POD'),
    f.col('valore').cast(DoubleType()).alias('ENERGIA_POW')
)
df_best_choice = df_best_choice.join(df_pow, on=['POD', 'UNIX_TIME'], how='left')

#################################
### CREAZIONE COLONNA CONSUMI ###
#################################

df_best_choice = df_best_choice.withColumn('IS_MISSING', f.when(
    (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_2G_BEST').isNull()),
    f.lit(1)
).otherwise(
    f.lit(0)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_1G_BEST_VALORE', f.col('ENERGIA_1G_BEST'))

df_best_choice = df_best_choice.withColumn('ENERGIA_1G_GIORNO_VALORE', f.when(
    (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNotNull()),
    f.col('ENERGIA_1G_GIORNO')
).otherwise(
    f.lit(None).cast(DoubleType())
))

df_best_choice = df_best_choice.withColumn('ENERGIA_2G_BEST_VALORE', f.when(
    (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_2G_BEST').isNotNull()),
    f.col('ENERGIA_2G_BEST')
).otherwise(
    f.lit(None).cast(DoubleType())
))

df_best_choice = df_best_choice.withColumn('ENERGIA_NA_VALORE', f.when(
    f.col('IS_MISSING') == 1,
    f.coalesce(f.col('ENERGIA_POW'), f.col('ENERGIA_1G_RECUPERO_NA'), f.col('ENERGIA_2G_RECUPERO_NA'))
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_EAC_VALORE', f.when(
    (f.col('IS_MISSING') == 1) & (f.col('ENERGIA_NA_VALORE').isNull()),
    f.col('EAC')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('CONSUMI', f.coalesce(
    f.col('ENERGIA_1G_BEST'),
    f.col('ENERGIA_1G_GIORNO'),
    f.col('ENERGIA_2G_BEST'),
    f.col('ENERGIA_POW'),
    f.col('ENERGIA_1G_RECUPERO_NA'),
    f.col('ENERGIA_2G_RECUPERO_NA'),
    f.col('EAC')
))

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - VALORI\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

df_best_choice.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC04DAVIDE')

############################
### AGGREGAZIONE CLUSTER ###
############################

if IS_PREVISIONE:

    NOME_TRANSFORM = 'Benchmark - Like Day'

    if TRACE_ANALISI:
        outpath_trace = f's3://{NOME_BUCKET}/datamodel/FORECAST_SAGEMAKER_ANALISI'
        data_corrente = INIZIO_PERIODO
        while data_corrente <= FINE_PERIODO:
            delete_folder_s3(bucket=NOME_BUCKET,
                             folder_path=f'datamodel/FORECAST_SAGEMAKER_ANALISI/SIMULAZIONE={COD_SIMULAZIONE}/NOME_TRANSFORM={NOME_TRANSFORM}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
            data_corrente += relativedelta(days=1)
        del data_corrente
        df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE)) \
            .withColumn('PRIMO_GIORNO_PREVISIONE', f.lit(INIZIO_PERIODO)) \
            .write.mode('append').partitionBy(['SIMULAZIONE', 'NOME_TRANSFORM', 'DATA']).parquet(outpath_trace)

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - TRACE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.sum('ENERGIA_POW_VALORE').alias('PREVISIONE_POW'),
        f.sum('ENERGIA_NA_VALORE').alias('PREVISIONE_NA_RECUPERATI'),
        f.sum('ENERGIA_EAC_VALORE').alias('PREVISIONE_NA_EAC'),
        f.sum('CONSUMI').alias('PREVISIONE_NO_KAPPA'),
        f.sum(f.col('CONSUMI')*f.col('KAPPA_PERDITE')).alias('PREVISIONE')
    )
    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
    df_best_choice = df_best_choice.withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM))
    df_best_choice = df_best_choice.withColumn('PRIMO_GIORNO_PREVISIONE', f.lit(INIZIO_PERIODO))

    # rewrite_table = False
    # if path_exists_s3(bucket=NOME_BUCKET, path=f'datamodel/FORECAST_SAGEMAKER/SIMULAZIONE={COD_SIMULAZIONE}/NOME_TRANSFORM={NOME_TRANSFORM}/', is_file=False):
    #     df_clean = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/FORECAST_SAGEMAKER/SIMULAZIONE={COD_SIMULAZIONE}/NOME_TRANSFORM={NOME_TRANSFORM}')
    #     df_clean = df_clean.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))
    #     if df_clean.count() > 0:
    #         df_clean = df_clean.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
    #         df_clean = df_clean.withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM))
    #         df_clean = df_clean.filter(f.col('PRIMO_GIORNO_PREVISIONE') != INIZIO_PERIODO)
    #         df_clean.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/STAGING_OVERWRITE_PREVISIONE_{INIZIO_PERIODO.strftime("%Y%m%d")}')
    #         rewrite_table = True
    #     del df_clean

    data_corrente = INIZIO_PERIODO
    output_table = f's3://{NOME_BUCKET}/datamodel/FORECAST_SAGEMAKER'
    while data_corrente <= FINE_PERIODO:
        delete_folder_s3(bucket=NOME_BUCKET,
                         folder_path=f'datamodel/FORECAST_SAGEMAKER/SIMULAZIONE={COD_SIMULAZIONE}/NOME_TRANSFORM={NOME_TRANSFORM}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
        data_corrente += relativedelta(days=1)
    del data_corrente

    # if rewrite_table:
    #     df_clean = spark.read.parquet(f's3://{NOME_BUCKET}/STAGING_OVERWRITE_PREVISIONE_{INIZIO_PERIODO.strftime("%Y%m%d")}')
    #     df_clean.write.mode('append').partitionBy(['SIMULAZIONE', 'NOME_TRANSFORM', 'DATA']).parquet(output_table)
    #     del df_clean

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - SVUOTAMENTO COMPLETATO\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'NOME_TRANSFORM', 'DATA']).parquet(output_table)

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - SCRITTURA COMPLETATA\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

else:

    if TRACE_ANALISI:
        outpath_trace = f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}_ANALISI'
        data_corrente = INIZIO_PERIODO
        while data_corrente <= FINE_PERIODO:
            delete_folder_s3(bucket=NOME_BUCKET,
                             folder_path=f'datamodel/{NOME_TABELLA}_ANALISI/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
            data_corrente += relativedelta(days=1)
        del data_corrente
        df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE)).write.mode('append').partitionBy(
            ['SIMULAZIONE', 'DATA']).parquet(outpath_trace)

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - TRACE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.count(f.lit(1)).cast('bigint').alias('N_POD'),
        f.sum('IS_MISSING').cast('bigint').alias('N_POD_MANCANTI'),
        f.sum('ENERGIA_1G_BEST_VALORE').alias('CONSUMI_1G_CERT'),
        f.sum('ENERGIA_1G_GIORNO_VALORE').alias('CONSUMI_1G_NOCERT'),
        f.sum('ENERGIA_2G_BEST_VALORE').alias('CONSUMI_2G_CERT'),
        f.sum('ENERGIA_POW').alias('CONSUMI_POW'),
        f.sum('ENERGIA_NA_VALORE').alias('CONSUMI_NA_RECUPERATI'),
        f.sum('ENERGIA_EAC_VALORE').alias('CONSUMI_NA_EAC'),
        f.sum('CONSUMI').alias('CONSUMI_NO_KAPPA'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('CONSUMI')).alias('CONSUMI'),
        f.sum('ENERGIA_1G_BEST').alias('THOR_1G_CERT'),
        f.sum('ENERGIA_1G_GIORNO').alias('THOR_1G_NOCERT'),
        f.sum('ENERGIA_2G_BEST').alias('THOR_2G_CERT')
    )

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

    data_corrente = INIZIO_PERIODO
    output_table = f's3://{NOME_BUCKET}/{tab_path}'
    while data_corrente <= FINE_PERIODO:
        delete_folder_s3(bucket=NOME_BUCKET,
                         folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
        data_corrente += relativedelta(days=1)
    del data_corrente

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - SVUOTAMENTO COMPLETATO\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - SCRITTURA COMPLETATA\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

