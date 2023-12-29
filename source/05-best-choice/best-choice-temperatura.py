from shared_lib import apply_map, melt, get_subfolders_s3, delete_folder_s3
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame as sparkDF
from pyspark.sql import functions as f
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
from datetime import datetime, date
import pandas as pd
import holidays
import calendar
import sys
import pickle
import boto3

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)
install_and_import('redshift_connector')

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
DATA_BACKTEST = datetime.strptime(sys.argv[7], '%Y-%m-%d') if IS_PREVISIONE else datetime.strptime(sys.argv[4], '%Y-%m-%d')+relativedelta(days=10)

if FINE_PERIODO < INIZIO_PERIODO:
    raise Exception('ERRORE: data fine periodo minore di data inizio periodo')

DATE_RANGE = [data.to_pydatetime().date() for data in pd.date_range(start=INIZIO_PERIODO, end=FINE_PERIODO, freq='D')]

LOG_PATH_FILE = f'logs/BEST_CHOICE/COD_SIMULAZIONE={COD_SIMULAZIONE}/PERIODO={INIZIO_PERIODO.strftime("%Y%m%d")}_{FINE_PERIODO.strftime("%Y%m%d")}/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log'
NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-prod-s3-model'

tab_path = f'datamodel/{NOME_TABELLA}/'

log_string = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - INIZIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

log_string += f"""{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - PARAMETRI INPUT:
    {COD_SIMULAZIONE}  
    {TRACE_ANALISI}  
    {INIZIO_PERIODO}  
    {FINE_PERIODO}  
    {EFFETTURARE_RECUPERO}  
    {IS_PREVISIONE}  
    {DATA_BACKTEST}
"""
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CARICO SESSIONE SPARK\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

if TRACE_ANALISI:
    sys_date_trace = datetime.now().strftime('%Y-%m-%d %H %M %S')

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

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
    df_ponti = spark.createDataFrame(LISTA_FESTIVITA, schema=schema_ponti)
    if df_ponti.groupBy('DATA', 'ZONA_FESTIVITA').count().filter(f.col('count') > 1).count() > 0:
        df_ponti.groupBy('DATA', 'ZONA_FESTIVITA').count().filter(f.col('count') > 1).show(100,False)
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

    df_best_choice = df_best_choice.withColumn('FESTIVITA', f.coalesce(f.col('FESTIVITA_PROVINCIA'), f.col('FESTIVITA_ITALIA')))\
        .drop('FESTIVITA_ITALIA', 'FESTIVITA_PROVINCIA')

    return df_best_choice


eval_array_udf = f.udf(lambda c: None if (c is None) else eval(c), ArrayType(StringType()))

################################
### CONFIGURAZIONE FESTIVITA ###
################################

df_lista_festivita = spark.read.option('delimiter', ';').option('header', True).option('inferSchema', False).csv(f's3://{NOME_BUCKET}/datainput/FESTIVITA')
df_lista_festivita = df_lista_festivita.select(
    f.to_date(f.col('DATA'), 'yyyy-MM-dd').alias('DATA'),
    f.col('TAG'),
    f.col('ZONA')
)

LISTA_FESTIVITA = df_lista_festivita.toPandas().values.tolist()

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - LISTA_FESTIVITA\n'
log_string += f'{LISTA_FESTIVITA}\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

del df_lista_festivita

################################
### COEFFICIENTE DISPERSIONI ###
################################

df_dispersioni = spark.read.option('delimiter', ';').option('header', True).option('inferSchema', False).csv(f's3://{NOME_BUCKET}/datainput/COEFFICIENTE_DISPERSIONI')
df_dispersioni = df_dispersioni.select(
    f.col('TENSIONE'),
    f.col('VALORE').cast(DoubleType()).alias('VALORE')
)
df_dispersioni = df_dispersioni.toPandas()

map_tensioni = df_dispersioni.set_index('TENSIONE')['VALORE'].to_dict()

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - COEFFICIENTE DISPERSIONI\n'
log_string += f'{map_tensioni}\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

del df_dispersioni

####################
### MAPPING AREA ###
####################

df_mapping_area = spark.read.option('delimiter', ';').option('header', True).option('inferSchema', False).csv(f's3://{NOME_BUCKET}/datainput/MAPPING_AREA')
df_mapping_area = df_mapping_area.select(
    f.col('AREA'),
    f.col('DECODIFICA')
)
df_mapping_area = df_mapping_area.toPandas()

map_area = df_mapping_area.set_index('AREA')['DECODIFICA'].to_dict()

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - MAPPING AREA\n'
log_string += f'{map_area}\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

del df_mapping_area

##############################
### CONFIGURAZIONE CLUSTER ###
##############################

df_conf_cluster = spark.read.option('delimiter', ';').option('header', True).option('inferSchema', False).csv(f's3://{NOME_BUCKET}/datainput/CONFIGURAZIONE_CLUSTER/')
df_conf_cluster = df_conf_cluster.select(
    f.col('SIMULAZIONE'),
    f.col('REGOLA_SELECT'),
    eval_array_udf(f.col('LISTA_POD_SINGOLI')).alias('LISTA_POD_SINGOLI'),
    eval_array_udf(f.col('CLUSTER_POD_SINGOLI')).alias('CLUSTER_POD_SINGOLI'),
    f.col('CLUSTER_POD_SINGOLI_NOME'),
    f.col('REGOLA_WHERE'),
    f.col('ORDINAMENTO').cast(IntegerType()).alias('ORDINAMENTO'),
    f.to_timestamp(f.col('VALIDITA'), 'yyyy-MM-dd HH:mm:SS').alias('VALIDITA')
)

df_conf_cluster = df_conf_cluster.filter(f.col('SIMULAZIONE') == COD_SIMULAZIONE)

try:
    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - LETTURA CONFIGURAZIONE CLUSTER\n'
    log_string += f'{df_conf_cluster.collect()}\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)
except:
    raise Exception('ERRORE: fallito il parsing del file CONFIGURAZIONE_CLUSTER')

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

COUNT_ANAGRAFICA_VALIDA = 100000

NOME_BUCKET_ANAGRAFICA = 'eva-prod-s3-datalake'
nome_tabella_anagrafica = 'root/EDP/Eva/Replication/PortafoglioCertificato/output/'

folder_anagrafica = get_subfolders_s3(bucket=NOME_BUCKET_ANAGRAFICA, path=nome_tabella_anagrafica)
folder_anagrafica = max([x.split('/')[-2] for x in folder_anagrafica])

df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET_ANAGRAFICA}/{nome_tabella_anagrafica}/{folder_anagrafica}')
df_anagrafica = df_anagrafica.filter(f.col('TRATTAMENTO') == 'O')

df_anagrafica = apply_map(
    df=df_anagrafica,
    col_name='id_area_gestionale',
    map_dict=map_area
)

df_anagrafica = df_anagrafica.withColumn('TIPO_FLUSSO',
                                         f.when(f.col('TIPO_MISURATORE') == 'G', f.lit('2G')).otherwise(
                                             f.lit('1G')))
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INIZIO_VALIDITA'), 'yyyy-MM-dd'))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FINE_VALIDITA'), 'yyyy-MM-dd'))
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_INI_VALIDITA'))))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_FIN_VALIDITA'))))

## ZONA - PROVINCIA ##

df_zona_provincia = spark.read.parquet(f's3://{NOME_BUCKET}/datainput/associazione_prov_zona.parquet')
df_zona_provincia = df_zona_provincia.select(
    f.col('provincia_crm').alias('PROVINCIA_CRM'),
    f.col('zona').alias('ZONA_RECUPERO')
)
df_anagrafica = df_anagrafica.join(df_zona_provincia, on=['PROVINCIA_CRM'], how='left')
df_anagrafica = df_anagrafica.withColumn('ZONA', f.coalesce(f.col('ZONA'), f.col('ZONA_RECUPERO')))

filtro_anag_date_validita = (f.col('DT_INI_VALIDITA') <= FINE_PERIODO) & (f.col('DT_FIN_VALIDITA') >= INIZIO_PERIODO)
last_validita = datetime(now.year, now.month, calendar.monthrange(now.year, now.month)[1]).date()

MAX_ITER = 60
while (df_anagrafica.filter((f.col('DT_INI_VALIDITA') <= last_validita) & (f.col('DT_FIN_VALIDITA') >= last_validita)).count() <= COUNT_ANAGRAFICA_VALIDA) and (MAX_ITER >= 0):
    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ANAGRAFICA NON TROVATA - {last_validita.strptime("%Y-%m-%d")}\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)
    last_validita += relativedelta(days=-1)
    MAX_ITER -= 1

if MAX_ITER <= 0:
    raise Exception('ERRORE: ricerca di data con anagrafica valida fallita')

df_anagrafica = df_anagrafica.cache()

mappa_date_anagrafica = dict()
for data_check in DATE_RANGE:
    if df_anagrafica.filter((f.col('DT_INI_VALIDITA') <= data_check) & (f.col('DT_FIN_VALIDITA') >= data_check)).count() > COUNT_ANAGRAFICA_VALIDA:
        mappa_date_anagrafica[data_check] = data_check
    else:
        mappa_date_anagrafica[data_check] = last_validita

df_anagrafica = df_anagrafica.filter(
    (f.col('DT_INI_VALIDITA') <= max(mappa_date_anagrafica.values())) & (f.col('DT_FIN_VALIDITA') >= min(mappa_date_anagrafica.values()))
)

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {mappa_date_anagrafica}\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ANAGRAFICA\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

# TODO: non dovrebbe servire
df_anagrafica = df_anagrafica.dropDuplicates(subset=['pod', 'DT_INI_VALIDITA', 'DT_FIN_VALIDITA'])

if TRACE_ANALISI:
    df_anagrafica.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace)) \
     .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/01_ANAGRAFICA')

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
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(lista_pod_singoli), concat_syntax).otherwise(f.col('CLUSTER')))
    elif cluster_pod_singoli is not None:
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(f.col('POD').isin(cluster_pod_singoli), f.lit(cluster_pod_singoli_nome)).otherwise(f.col('CLUSTER')))
    else:
        if not pd.isna(regola_where):
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(f'CASE WHEN {regola_where} THEN {regola_select} ELSE CLUSTER END'))
        else:
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(regola_select))

df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.coalesce(f.col('CLUSTER'), f.lit('CLUSTER_MANCANTI')))

df_anagrafica = apply_map(
    df=df_anagrafica,
    col_name='TENSIONE',
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

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CLUSTER\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

df_anagrafica = df_anagrafica.cache()

if TRACE_ANALISI:
    df_anagrafica.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/02_CLUSTER')

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

if IS_PREVISIONE and (FINE_PERIODO > now):
    df_calendario = apply_map(
        df=df_calendario,
        col_name='DATA',
        map_dict=mappa_date_anagrafica,
        output_col_name='DATA_ANAGRAFICA'
    )
else:
    df_calendario = df_calendario.withColumn('DATA_ANAGRAFICA', f.col('DATA'))

df_best_choice = df_anagrafica.crossJoin(df_calendario)

df_best_choice = df_best_choice.filter(f.col('DATA_ANAGRAFICA').between(f.col('DT_INI_VALIDITA'), f.col('DT_FIN_VALIDITA')))

df_best_choice = df_best_choice.drop('DATA_ANAGRAFICA')

df_best_choice = add_festivita(spark, df_best_choice)

df_num_giorni_anno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
    .select('ANNO', 'DATA').distinct().groupBy('ANNO').count().withColumnRenamed('count', 'NUM_SETTIMANE')
df_num_giorni_anno = df_num_giorni_anno.withColumn('NUM_SETTIMANE', (f.col('NUM_SETTIMANE')/f.lit(7)).cast(DoubleType()))
df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
df_best_choice = df_best_choice.withColumn('EAC', f.col('CONSUMO_ANNUO') / f.col('NUM_SETTIMANE')) \
    .drop('NUM_SETTIMANE', 'ANNO')

## PROFILI EAC ##

df_profili_eac = spark.read.parquet(f's3://{NOME_BUCKET}/datainput/profili_eac.parquet')
df_profili_eac = df_profili_eac.select(
    f.col('GIORNO_SETTIMANA'),
    f.col('ORA_GME'),
    f.col('PERCENTUALE')
)
df_best_choice = df_best_choice.join(df_profili_eac, on=['GIORNO_SETTIMANA', 'ORA_GME'], how='left')
df_best_choice = df_best_choice.withColumn('PERCENTUALE', f.coalesce(f.col('PERCENTUALE'), f.lit(1/(7*24))))
df_best_choice = df_best_choice.withColumn('EAC', f.col('EAC')*f.col('PERCENTUALE')).drop('PERCENTUALE')

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - CALENDARIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

if TRACE_ANALISI:
    df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/03_CALENDARIO')

###########################
### AGGIUSTIAMO I CAMPI ###
###########################

# MISURE 1G - GIORNO #

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ESTRAZIONE MISURE 1G GIORNO - {INIZIO_PERIODO} - {FINE_PERIODO} - {DATA_BACKTEST} \n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

df_1g_giorno = extract_1g_giorno(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)

if TRACE_ANALISI:
    df_1g_giorno.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/MISURE_1G_GIORNO')

df_best_choice = df_best_choice.join(df_1g_giorno, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_1g_giorno

# MISURE 1G - BEST #

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ESTRAZIONE MISURE 1G BEST - {INIZIO_PERIODO} - {FINE_PERIODO} - {DATA_BACKTEST} \n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

df_1g_best = extract_1g_best(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)

if TRACE_ANALISI:
    df_1g_best.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/MISURE_1G_BEST')

df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_1g_best

# MISURE 2G - BEST #

log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ESTRAZIONE MISURE 2G BEST - {INIZIO_PERIODO} - {FINE_PERIODO} - {DATA_BACKTEST} \n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

df_2g_best = extract_2g_best(spark, INIZIO_PERIODO, FINE_PERIODO, DATA_BACKTEST)

if TRACE_ANALISI:
    df_2g_best.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/MISURE_2G')

df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
del df_2g_best

df_best_choice = df_best_choice.cache()

if TRACE_ANALISI:
    df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/04_MISURE')

if EFFETTURARE_RECUPERO:

    DELTA_GG_1G = 60
    DELTA_GG_2G = 15

    df_festivita_da_recuperare = get_ponti(spark).filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO)).select('FESTIVITA').distinct()
    DATE_AGGIUNTIVE = [x.DATA for x in get_ponti(spark).filter(f.col('DATA') < INIZIO_PERIODO).join(df_festivita_da_recuperare, on=['FESTIVITA'], how='inner').select('DATA').distinct().collect()]
    del df_festivita_da_recuperare

    make_chiave_recupero = f.coalesce(f.col('FESTIVITA'), f.col('GIORNO_SETTIMANA'))

    df_best_choice = df_best_choice.withColumn('CHIAVE_RECUPERO', make_chiave_recupero)

    df_anagrafica = df_best_choice.select('POD', 'PROVINCIA_CRM').distinct()

    wnd_recupero = Window.partitionBy('POD', 'ORA_GME', 'CHIAVE_RECUPERO').orderBy(f.col('DATA').desc())

    df_1g_best = extract_1g_best(
        spark,
        inizio_periodo=min(now, INIZIO_PERIODO)-relativedelta(days=DELTA_GG_1G),
        fine_periodo=min(now, FINE_PERIODO),
        data_backtest=DATA_BACKTEST,
        date_recupero_singole=tuple(DATE_AGGIUNTIVE)
    )
    df_1g_best = df_1g_best.withColumnRenamed('ENERGIA_1G_BEST', 'ENERGIA_1G_RECUPERO_NA')
    df_1g_best = df_1g_best.join(df_anagrafica, on=['POD'], how='inner')
    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
        .select('DATA', 'GIORNO_SETTIMANA').distinct()
    df_1g_best = df_1g_best.join(df_calendario, on=['DATA'], how='left')
    df_1g_best = add_festivita(spark, df_1g_best)
    df_1g_best = df_1g_best.withColumn('CHIAVE_RECUPERO', make_chiave_recupero)
    df_1g_best = df_1g_best.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1)
    df_1g_best = df_1g_best.select(
        f.col('POD'),
        f.col('ORA_GME'),
        f.col('CHIAVE_RECUPERO'),
        f.col('ENERGIA_1G_RECUPERO_NA'),
        f.col('DATA').alias('DATA_RECUPERO_1G')
    )
    df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'ORA_GME', 'CHIAVE_RECUPERO'], how='left')

    if TRACE_ANALISI:
        df_1g_best.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/RECUPERO_1G')

    del df_1g_best, df_calendario

    df_2g_best = extract_2g_best(
        spark,
        inizio_periodo=min(now, INIZIO_PERIODO)-relativedelta(days=DELTA_GG_2G),
        fine_periodo=min(now, FINE_PERIODO),
        data_backtest=DATA_BACKTEST,
        date_recupero_singole=tuple(DATE_AGGIUNTIVE)
    )
    df_2g_best = df_2g_best.withColumnRenamed('ENERGIA_2G_BEST', 'ENERGIA_2G_RECUPERO_NA')
    df_2g_best = df_2g_best.join(df_anagrafica, on=['POD'], how='inner')
    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME/') \
        .select('DATA', 'GIORNO_SETTIMANA').distinct()
    df_2g_best = df_2g_best.join(df_calendario, on=['DATA'], how='left')
    df_2g_best = add_festivita(spark, df_2g_best)
    df_2g_best = df_2g_best.withColumn('CHIAVE_RECUPERO', make_chiave_recupero)
    df_2g_best = df_2g_best.withColumn('RANK', f.row_number().over(wnd_recupero)).filter(f.col('RANK') == 1)
    df_2g_best = df_2g_best.select(
        f.col('POD'),
        f.col('ORA_GME'),
        f.col('CHIAVE_RECUPERO'),
        f.col('ENERGIA_2G_RECUPERO_NA'),
        f.col('DATA').alias('DATA_RECUPERO_2G')
    )
    df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'ORA_GME', 'CHIAVE_RECUPERO'], how='left')

    if TRACE_ANALISI:
        df_2g_best.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/RECUPERO_2G')

    del df_2g_best, df_calendario

else:
    df_best_choice = df_best_choice.withColumn('DATA_RECUPERO_1G', f.lit(None).cast(DateType()))
    df_best_choice = df_best_choice.withColumn('DATA_RECUPERO_2G', f.lit(None).cast(DateType()))
    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_RECUPERO_NA', f.lit(None).cast(DoubleType()))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_RECUPERO_NA', f.lit(None).cast(DoubleType()))

if TRACE_ANALISI:
    df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/05_RECUPERO')

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

if TRACE_ANALISI:
    df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
        .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/06_POW')

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
    (f.col('IS_MISSING') == 1) & (f.col('ENERGIA_POW').isNull()),
    f.coalesce(f.col('ENERGIA_1G_RECUPERO_NA'), f.col('ENERGIA_2G_RECUPERO_NA'))
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
            .withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM)) \
            .write.mode('append').partitionBy(['SIMULAZIONE', 'NOME_TRANSFORM', 'DATA']).parquet(outpath_trace)

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - TRACE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.count(f.lit(1)).cast('bigint').alias('N_POD'),
        f.sum(f.col('ENERGIA_POW')*f.col('KAPPA_PERDITE')).alias('PREVISIONE_POW'),
        f.sum(f.col('ENERGIA_NA_VALORE')*f.col('KAPPA_PERDITE')).alias('PREVISIONE_NA_RECUPERATI'),
        f.sum(f.col('ENERGIA_EAC_VALORE')*f.col('KAPPA_PERDITE')).alias('PREVISIONE_NA_EAC'),
        f.sum(f.col('CONSUMI')*f.col('KAPPA_PERDITE')).alias('PREVISIONE'),
        f.collect_set(f.col('DATA_RECUPERO_1G')).alias('DATE_RECUPERO_1G'),
        f.collect_set(f.col('DATA_RECUPERO_2G')).alias('DATE_RECUPERO_2G')
    ).cache()

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
    df_best_choice = df_best_choice.withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM))
    df_best_choice = df_best_choice.withColumn('PRIMO_GIORNO_PREVISIONE', f.lit(INIZIO_PERIODO))

    df_best_choice = df_best_choice.withColumn('ZONA', f.split(f.col('CLUSTER'), '#')[0])
    df_best_choice = df_best_choice.withColumn('ID_AREA_GESTIONALE', f.split(f.col('CLUSTER'), '#')[1])
    df_best_choice = df_best_choice.withColumn('PROVINCIA', f.split(f.col('CLUSTER'), '#')[3])

    df_best_choice = df_best_choice.withColumn('DATA_OLD', f.date_sub(f.col('DATA'), 7))

    if TRACE_ANALISI:
        df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/07_PREVISIONE_AGGREGAZIONE')

    query = f"""
        SELECT * FROM 
            mdp_slv.f_meteo_epson
        WHERE
            data - flow_date = 1 and
            flow_id = 'EPSON_TEMPERATURES_FCT'
    """

    df_comuni = spark.read.parquet(f's3://{NOME_BUCKET}/datainput/decodifica_cod_provincia.parquet')

    df_comuni = df_comuni.select(
        'COD_PROVINCIA',
        'PROVINCIA'
    ).distinct()

    conn = redshift_connector.connect(
        host='ewfpamdpdr01.corp.awsedison.it',
        database='mdpprodred',
        port=5439,
        user='mdp_red_eva_prod',
        password='B0ZCaAg2OZYX8jFgK5oo'
    )
    cursor = conn.cursor()
    cursor.execute("select * from mdp_slv.f_meteo_epson WHERE data - flow_date = 1 and flow_id = 'EPSON_TEMPERATURES_FCT'")
    df_temperature: pd.DataFrame = cursor.fetch_dataframe()[['data', 'cod_provincia', 'temperature_avg']]
    df_temperature = spark.createDataFrame(df_temperature)

    df_temperature = df_temperature.groupBy(
        f.col('data').alias('DATA'),
        f.col('cod_provincia').alias('COD_PROVINCIA')
    ).agg(
        f.avg(f.col('temperature_avg')).alias('TEMP_AVG')
    )

    df_temperature = df_temperature.join(df_comuni, on=['COD_PROVINCIA'], how='inner')

    if TRACE_ANALISI:
        df_temperature.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/08_PREVISIONE_TEMPERATURA')

    df_temperature = df_temperature.drop('COD_PROVINCIA')

    df_best_choice = df_best_choice.join(
        df_temperature,
        on=['DATA', 'PROVINCIA'],
        how='left'
    )
    df_best_choice = df_best_choice.join(
        df_temperature.withColumnRenamed('DATA', 'DATA_OLD').withColumnRenamed('TEMP_AVG', 'TEMP_AVG_OLD'),
        on=['DATA_OLD', 'PROVINCIA'],
        how='left'
    )

    if TRACE_ANALISI:
        df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/09_PREVISIONE_BEST_CHOICE_TEMPERATURA')

    df_anag = spark.read.parquet(f's3://{NOME_BUCKET_ANAGRAFICA}/{nome_tabella_anagrafica}/{folder_anagrafica}')
    df_anag = df_anag.filter(f.col('TRATTAMENTO') == 'O')
    df_anag = df_anag.withColumn('DT_INI_VALIDITA',f.to_timestamp(f.col('DT_INIZIO_VALIDITA'), 'yyyy-MM-dd'))
    df_anag = df_anag.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FINE_VALIDITA'), 'yyyy-MM-dd'))
    df_anag = df_anag.withColumn('DT_INI_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_INI_VALIDITA'))))
    df_anag = df_anag.withColumn('DT_FIN_VALIDITA', f.to_date(f.date_trunc('day', f.col('DT_FIN_VALIDITA')))).cache()

    log_string += f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - LEGGO ANAGRAFICA PER PREVISIONE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key=LOG_PATH_FILE)

    map_npod = dict()
    map_npod_old = dict()

    df_npod = spark.createDataFrame(
        spark.sparkContext.emptyRDD(),
        schema=StructType([
            StructField('DATA', DateType(), True),
            StructField('PROVINCIA', DateType(), True),
            StructField('NPOD_NEW', IntegerType(), True),
        ])
    )

    df_npod_old = spark.createDataFrame(
        spark.sparkContext.emptyRDD(),
        schema=StructType([
            StructField('DATA_OLD', DateType(), True),
            StructField('PROVINCIA', DateType(), True),
            StructField('NPOD_OLD', IntegerType(), True),
        ])
    )

    for data_curr in [x.DATA for x in df_best_choice.select('DATA').distinct().collect()]:
        if df_anag.filter((f.col('DT_INI_VALIDITA') <= data_curr) & (f.col('DT_FIN_VALIDITA') >= data_curr)).count() <= COUNT_ANAGRAFICA_VALIDA:
            df_anag_corrente = df_anag.filter((f.col('DT_INI_VALIDITA') <= data_curr) & (f.col('DT_FIN_VALIDITA') >= data_curr))
        else:
            df_anag_corrente = df_anag.filter((f.col('DT_INI_VALIDITA') <= last_validita) & (f.col('DT_FIN_VALIDITA') >= last_validita))
        df_anag_corrente = df_anag_corrente.groupBy(f.col('PROVINCIA_CRM').alias('PROVINCIA')).agg(f.countDistinct(f.col('POD')).alias('NPOD_NEW'))
        df_anag_corrente = df_anag_corrente.withColumn('DATA', f.lit(data_curr))
        df_npod = df_npod.unionByName(df_anag_corrente)

    if TRACE_ANALISI:
        df_npod.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/10_PREVISIONE_NPOD')

    for data_curr in [x.DATA_OLD for x in df_best_choice.select('DATA_OLD').distinct().collect()]:
        if df_anag.filter((f.col('DT_INI_VALIDITA') <= data_curr) & (f.col('DT_FIN_VALIDITA') >= data_curr)).count() <= COUNT_ANAGRAFICA_VALIDA:
            df_anag_corrente = df_anag.filter((f.col('DT_INI_VALIDITA') <= data_curr) & (f.col('DT_FIN_VALIDITA') >= data_curr))
        else:
            df_anag_corrente = df_anag.filter((f.col('DT_INI_VALIDITA') <= last_validita) & (f.col('DT_FIN_VALIDITA') >= last_validita))
        df_anag_corrente = df_anag_corrente.groupBy(f.col('PROVINCIA_CRM').alias('PROVINCIA')).agg(f.countDistinct(f.col('POD')).alias('NPOD_OLD'))
        df_anag_corrente = df_anag_corrente.withColumn('DATA_OLD', f.lit(data_curr))
        df_npod_old = df_npod_old.unionByName(df_anag_corrente)

    if TRACE_ANALISI:
        df_npod_old.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/11_PREVISIONE_NPOD_OLD')

    df_best_choice = df_best_choice.join(df_npod, on=['DATA', 'PROVINCIA'], how='left')
    df_best_choice = df_best_choice.join(df_npod_old, on=['DATA_OLD', 'PROVINCIA'], how='left')

    if TRACE_ANALISI:
        df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/12_PREVISIONE_BEST_CHOICE_POD')

    ### REGRESSIONE - CHARLIE ###

    s3 = boto3.resource('s3')
    input = pickle.loads(s3.Bucket(NOME_BUCKET).Object("Temp-regression-result-charlie.obj").get()['Body'].read())[0]

    coeff = pd.DataFrame()

    for cluster, result, _ in input:
        result.name = 'COEFF'
        result.index.name = 'NOME_COEFF'
        result = result.to_frame()
        result['CLUSTER'] = cluster
        coeff = pd.concat([coeff, result])

    df_coeff = pd.DataFrame(index=pd.MultiIndex.from_product([coeff['CLUSTER'].unique(), range(24)]))

    coeff = coeff.reset_index().set_index(['CLUSTER', 'NOME_COEFF'])['COEFF']

    for cluster, nome_coeff in coeff.index:
        if nome_coeff == 'const':
            df_coeff.loc[(cluster, slice(None)), 'ALPHA_CONST'] = coeff[(cluster, nome_coeff)]
        elif nome_coeff == 'n_pod':
            df_coeff.loc[(cluster, slice(None)), 'ALPHA_NPOD'] = coeff[(cluster, nome_coeff)]
        else:
            if nome_coeff == 'temperature_sq':
                df_coeff.loc[(cluster, slice(None)), 'ALPHA_TEMP_SQ'] = coeff[(cluster, nome_coeff)]
            elif nome_coeff == 'temperature_avg':
                df_coeff.loc[(cluster, slice(None)), 'ALPHA_TEMP'] = coeff[(cluster, nome_coeff)]

    for cluster, _, rsquared in input:
        df_coeff.loc[(cluster, slice(None)), 'ALPHA_R2'] = rsquared

    df_coeff.index = df_coeff.index.set_names(['CLUSTER', 'ORA_GME'])
    df_coeff = df_coeff.reset_index()
    df_coeff['ORA_GME'] = df_coeff['ORA_GME'] + 1

    df_coeff = spark.createDataFrame(df_coeff)

    df_best_choice = df_best_choice.join(df_coeff, on=['CLUSTER', 'ORA_GME'], how='left')

    ### REGRESSIONE - DAVIDE ###

    s3 = boto3.resource('s3')
    input = pickle.loads(s3.Bucket(NOME_BUCKET).Object("Temp-regression-result.obj").get()['Body'].read())[0]

    coeff = pd.DataFrame()

    for cluster, result, _ in input:
        result.name = 'COEFF'
        result.index.name = 'NOME_COEFF'
        result = result.to_frame()
        result['CLUSTER'] = cluster
        coeff = pd.concat([coeff, result])

    df_coeff = pd.DataFrame(index=pd.MultiIndex.from_product([coeff['CLUSTER'].unique(), range(24)]))

    coeff = coeff.reset_index().set_index(['CLUSTER', 'NOME_COEFF'])['COEFF']

    for cluster, nome_coeff in coeff.index:
        if nome_coeff == 'const':
            df_coeff.loc[(cluster, slice(None)), 'BETA_CONST'] = coeff[(cluster, nome_coeff)]
        elif nome_coeff == 'n_pod':
            df_coeff.loc[(cluster, slice(None)), 'BETA_NPOD'] = coeff[(cluster, nome_coeff)]
        else:
            if ('_temp_' in nome_coeff) and ('_cube_' in nome_coeff):
                ora = int(nome_coeff[nome_coeff.find('temp_cube_') + len('temp_cube_'):])
                df_coeff.loc[(cluster, ora), 'BETA_TEMP_CUBE'] = coeff[(cluster, nome_coeff)]
            elif ('_temp_' in nome_coeff) and ('_sq_' in nome_coeff):
                ora = int(nome_coeff[nome_coeff.find('temp_sq_') + len('temp_sq_'):])
                df_coeff.loc[(cluster, ora), 'BETA_TEMP_SQ'] = coeff[(cluster, nome_coeff)]
            elif ('_temp_' in nome_coeff):
                ora = int(nome_coeff[nome_coeff.find('temp_') + len('temp_'):])
                df_coeff.loc[(cluster, ora), 'BETA_TEMP'] = coeff[(cluster, nome_coeff)]

    for cluster, _, rsquared in input:
        df_coeff.loc[(cluster, slice(None)), 'BETA_R2'] = rsquared

    df_coeff.index = df_coeff.index.set_names(['CLUSTER', 'ORA_GME'])
    df_coeff = df_coeff.reset_index()
    df_coeff['ORA_GME'] = df_coeff['ORA_GME'] + 1

    df_coeff = spark.createDataFrame(df_coeff)

    df_best_choice = df_best_choice.join(df_coeff, on=['CLUSTER', 'ORA_GME'], how='left')

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
    df_best_choice = df_best_choice.withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM))
    df_best_choice = df_best_choice.withColumn('PRIMO_GIORNO_PREVISIONE', f.lit(INIZIO_PERIODO))

    df_best_choice = df_best_choice.withColumn('DELTA_REGRESSIONE_ALPHA', f.expr(
        'ALPHA_R2*( ALPHA_TEMP*(TEMP_AVG-TEMP_AVG_OLD) + ALPHA_TEMP_SQ*(pow(TEMP_AVG,2)-pow(TEMP_AVG_OLD,2)) + ALPHA_NPOD*(NPOD_NEW-NPOD_OLD))'
    ))

    df_best_choice = df_best_choice.withColumn('DELTA_REGRESSIONE_BETA', f.expr(
        'BETA_R2*( BETA_TEMP*(TEMP_AVG-TEMP_AVG_OLD) + BETA_TEMP_SQ*(pow(TEMP_AVG,2)-pow(TEMP_AVG_OLD,2)) + BETA_TEMP_CUBE*(pow(TEMP_AVG,3)-pow(TEMP_AVG_OLD,3)) + BETA_NPOD*(NPOD_NEW-NPOD_OLD))'
    ))

    if TRACE_ANALISI:
        df_best_choice.withColumn('SYS_DATE_TRACE', f.lit(sys_date_trace))\
            .write.partitionBy('SYS_DATE_TRACE').mode('append').parquet(f's3://{NOME_BUCKET}/DEBUG/13_PREVISIONE_REGRESSIONE')

    #df_best_choice = df_best_choice.withColumn('PREVISIONE', f.col('PREVISIONE') + f.col('DELTA_REGRESSIONE')*map_tensioni['BT'])

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
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_1G_BEST_VALORE')).alias('CONSUMI_1G_CERT'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_1G_GIORNO_VALORE')).alias('CONSUMI_1G_NOCERT'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_2G_BEST_VALORE')).alias('CONSUMI_2G_CERT'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_POW')).alias('CONSUMI_POW'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_NA_VALORE')).alias('CONSUMI_NA_RECUPERATI'),
        f.sum(f.col('KAPPA_PERDITE')*f.col('ENERGIA_EAC_VALORE')).alias('CONSUMI_NA_EAC'),
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

