import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])

from pyspark.sql.types import *
from datetime import date, timedelta
from pyspark.sql import Window
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from typing import Iterable
from pyspark.sql.dataframe import DataFrame as sparkDF
import boto3
from dateutil.relativedelta import relativedelta
import time


def check_path_s3(path: str, is_file: bool = False):
    if len(path) > 0:
        if path[-1] != '/' and not is_file: path = f'{path}/'
        if path[0] == '/': path = path[1:]
    return path


def delete_folder_s3(bucket: str, folder_path: str):
    folder_path = check_path_s3(folder_path)
    boto3.resource('s3').Bucket(bucket).objects.filter(Prefix=folder_path).delete()


def melt(
        df: sparkDF,
        id_vars: Iterable[str], value_vars: Iterable[str],
        var_name: str = "variable", value_name: str = "value"
) -> sparkDF:
    _vars_and_vals = f.array(*(f.struct(f.lit(c).alias(var_name), f.col(c).alias(value_name)) for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals", f.explode(_vars_and_vals))
    cols = id_vars+[f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)


spark = SparkSession.builder.appName('BEST_CHOICE v1.0.0').enableHiveSupport().getOrCreate()

COD_SIMULAZIONE = '1000'
NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'

DELTA_GIORNI = 5

OVERWRITE_TABLE = True

data_inizio = date(2023, 1, 1)
data_fine = date(2023, 9, 24)
tab_path = f'datamodel/{NOME_TABELLA}/'

LISTA_DATE = list()
while data_inizio <= data_fine:
    data_limsup = min(data_inizio + relativedelta(days=DELTA_GIORNI) + relativedelta(days=-1), data_fine)
    LISTA_DATE.append([data_inizio, data_limsup])
    data_inizio = data_limsup + relativedelta(days=1)

if OVERWRITE_TABLE:
    delete_folder_s3(bucket=NOME_BUCKET, folder_path=tab_path)

for INIZIO_PERIODO, FINE_PERIODO in LISTA_DATE:

    NUM_GIORNI = (FINE_PERIODO - INIZIO_PERIODO).days + 1
    RANGE_DATE = [INIZIO_PERIODO + timedelta(days=x) for x in range(NUM_GIORNI)]
    RANGE_DATE_STRING = [f"'{x.strftime('%Y/%m/%d')}'" for x in RANGE_DATE]

    ##################
    ### ANAGRAFICA ###
    ##################

    # POD attivi in cui dobbiao calcolare la BEST CHOICE (DATA FORNITURA)
    df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII')

    #df_anagrafica = df_anagrafica.filter((f.col('TRATTAMENTO') != 'O') & (f.col('TRATTAMENTO') != 'I'))

    # Filtriamo solo i pod che hanno inizio e fine fornitura nel periodo di interesse
    df_anagrafica = df_anagrafica.filter(~(
            (f.col('DT_FIN_VALIDITA') < INIZIO_PERIODO) |
            (f.col('DT_INI_VALIDITA') > FINE_PERIODO)
    ))

    # TODO: vedere come gestire se ci sono pod multipli
    df_anagrafica = df_anagrafica.groupBy('POD').agg(
        f.avg(f.col('CONSUMO_ANNUO_COMPLESSIVO')).alias('EAC')
    )

    ###############
    ### CLUSTER ###
    ###############

    df_cluster = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CLUSTER/COD_SIMULAZIONE={COD_SIMULAZIONE}')

    df_anagrafica = df_anagrafica.join(df_cluster, on=['POD'], how='left')
    df_anagrafica = df_anagrafica.fillna('CLUSTER_MANCANTI', subset=['CLUSTER'])

    ######################
    ### CALENDARIO-POD ###
    ######################

    df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO)) \
        .select(
            'TIMESTAMP',
            'DATA',
            'ORA_GME'
    )

    df_num_giorni_anno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
        .groupBy('ANNO').count().withColumnRenamed('count', 'NUM_GIORNO_ORA')

    df_best_choice = df_anagrafica.crossJoin(df_calendario)
    df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
    df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
    df_best_choice = df_best_choice.withColumn('EAC_GIORNO_ORA', f.col('EAC')/f.col('NUM_GIORNO_ORA')) \
        .drop('NUM_GIORNO_ORA', 'ANNO')

    ##########################
    ### MISURE 1G - GIORNO ###
    ##########################

    df_1g_enelg = spark.sql(f"""
        SELECT * FROM 
            `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            ymd IN ({', '.join(RANGE_DATE_STRING)})  AND
            cd_flow = 'ENEL-G' AND
            grandezza = 'A'
    """)

    df_1g_enelg = df_1g_enelg.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_1g_enelg = df_1g_enelg.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    #TODO: eliminare drop_duplicates
    df_1g_enelg = df_1g_enelg.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_1g_enelg.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_1g_enelg.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_1g_enelg.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_1G')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 1g-giorno')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_1g_enelg = df_1g_enelg.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_1g_enelg = df_1g_enelg.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g_enelg = melt(
        df=df_1g_enelg,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_GIORNO'
    )

    df_1g_enelg = df_1g_enelg.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g_enelg = df_1g_enelg.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_1g_enelg, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_1g_enelg

    ########################
    ### MISURE 1G - BEST ###
    ########################

    df_1g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m1g AS CURVA
        WHERE 
            ymd IN ({', '.join(RANGE_DATE_STRING)}) AND
            grandezza = 'A'
    """)

    df_misura = spark.sql('SELECT * FROM `623333656140/thr_prod_glue_db`.ee_misura_m1g') \
        .select(
            'id_misura_m2g',
            'motivazione'
    )

    df_1g = df_1g.join(df_misura, on=['id_misura_m2g'], how='left')

    df_1g = df_1g.filter(f.col('motivazione') != '3')

    df_1g = df_1g.withColumn('RANK_ORIGINE',
        f.when(f.col('cd_flow') == 'XLSX', f.lit(7))
         .when(f.col('cd_flow') == 'RFO', f.lit(6))
         .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato')=='E') & (f.col('validato')=='S')), f.lit(5))
         .when(((f.col('cd_flow') == 'ENEL-M') | (f.col('cd_flow') == 'M15DL')), f.lit(4))
         .when(f.col('cd_flow') == 'ENEL-G', f.lit(3))
         .when(f.col('cd_flow') == 'SOS', f.lit(2))
         .when(((f.col('cd_flow') == 'PDO') & (f.col('tipodato')!='E') & (f.col('validato')!='S')), f.lit(1))
         .otherwise(f.lit(-1))
    )

    w_1g = Window.partitionBy('POD', 'YMD').orderBy(f.col('RANK_ORIGINE').desc(), f.col('ts').desc())
    df_1g = df_1g.withColumn('RANK', f.row_number().over(w_1g))

    df_1g = df_1g.filter(f.col('RANK') == 1)

    #TODO: eliminare drop_duplicates
    df_1g = df_1g.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_1g.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_1g.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_1g.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_1G_BEST')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 1g')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_1g = df_1g.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_1g = df_1g.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_1g = melt(
        df=df_1g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_1G_BEST'
    )

    df_1g = df_1g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_1g = df_1g.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_1g, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_1g

    ########################
    ### MISURE 2G - BEST ###
    ########################

    df_2g = spark.sql(f"""
        SELECT * FROM 
        `623333656140/thr_prod_glue_db`.ee_curva_m2g_bc AS CURVA
        WHERE 
            ymd IN ({', '.join(RANGE_DATE_STRING)}) AND
            grandezza = 'A' AND
            flg_best_choice = 'Y'
    """)

    df_2g = df_2g.withColumn('MAX_TS', f.max('TS').over(Window.partitionBy('POD', 'YMD')))
    df_2g = df_2g.filter(f.col('TS') == f.col('MAX_TS')).drop('MAX_TS')

    #TODO: eliminare drop_duplicates
    df_2g = df_2g.dropDuplicates(subset=['POD', 'YMD', 'TS'])

    #if df_2g.groupBy('POD', 'YMD').count().filter(f.col('count')>1).count() > 0:
    #    check = df_2g.groupBy('POD', 'YMD').count().filter(f.col('count') > 1).select('POD', 'YMD')
    #    df_2g.join(check, on=['POD', 'YMD'], how='inner').repartition(1).write.mode('overwrite').csv(f's3://{NOME_BUCKET}/datamodel/DUPLICATI_2G')
    #    raise AttributeError('ERRORE: POD, GIORNO non chiave in 2g')

    for hh in range(0, 25):
        sum_col = f.col(f'con_e_{(hh*4+1):03}')
        for qh_idx in range(2, 5):
            sum_col = sum_col + f.col(f'con_e_{(hh*4+qh_idx):03}')
        df_2g = df_2g.withColumn(f'con_e_{(hh+1):02}', sum_col)

    df_2g = df_2g.select(
        f.col('POD'),
        f.to_date(f.col('ymd'), 'yyyy/MM/dd').alias('DATA'),
        *[f.col(f'con_e_{(hh+1):02}').cast(DoubleType()) for hh in range(0, 25)]
    )

    df_2g = melt(
        df=df_2g,
        id_vars=['POD', 'DATA'],
        value_vars=[f'con_e_{(hh+1):02}' for hh in range(0, 25)],
        var_name='ORA',
        value_name='ENERGIA_2G_BEST'
    )

    df_2g = df_2g.withColumn('ORA', f.substring(f.col('ORA'), -2, 2).cast(IntegerType()))
    df_2g = df_2g.withColumnRenamed('ORA', 'ORA_GME')

    df_best_choice = df_best_choice.join(df_2g, on=['POD', 'DATA', 'ORA_GME'], how='left')

    del df_2g

    cond_1g = (f.col('ENERGIA_1G_GIORNO').isNotNull() | f.col('ENERGIA_1G_BEST').isNotNull()) & (f.col('ENERGIA_2G_BEST').isNull())
    cond_2g = (f.col('ENERGIA_1G_GIORNO').isNull() & f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNotNull())
    no_dato_cond = (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNull())

    df_best_choice = df_best_choice.withColumn('TIPOLOGIA_MISURA',
       f.when(cond_1g, f.lit('1G'))
       .when(cond_2g, f.lit('2G'))
       .when(no_dato_cond, f.lit('MANCANTE'))
       .otherwise(f.lit('MIX'))
    )

    ############################
    ### AGGREGAZIONE CLUSTER ###
    ############################

    no_dato_cond = (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_2G_BEST').isNull())

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'TIMESTAMP').agg(
        f.sum('EAC_GIORNO_ORA').alias('EAC'),
        f.sum(f.when(no_dato_cond, f.col('EAC_GIORNO_ORA')).otherwise(f.lit(0.))).alias('EAC_MANCANTE'),
        f.count(f.lit(1)).alias('N_POD'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA') == 'MANCANTE', f.lit(1)).otherwise(f.lit(0))).alias('N_POD_MANCANTI'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA') == '1G', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_1G'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA') == '2G', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_2G'),
        f.sum(f.when(f.col('TIPOLOGIA_MISURA') == 'MIX', f.lit(1)).otherwise(f.lit(0))).alias('N_MISURE_MIX'),
        f.sum('ENERGIA_1G_GIORNO').alias('ENERGIA_1G_GIORNO'),
        f.sum('ENERGIA_1G_BEST').alias('ENERGIA_1G_BEST'),
        f.sum('ENERGIA_2G_BEST').alias('ENERGIA_2G_BEST')
    )

    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(f's3://{NOME_BUCKET}/{tab_path}')

####################################
### AGGIUNTA TABELLA SU SPECTRUM ###
####################################

create_string = f"""
create external table if not exists datamodel.eva_qa_glue_meter_consuntivo(
    CLUSTER varchar,
    TIMESTAMP timestamp,
    EAC double precision,
    EAC_MANCANTE double precision,
    N_POD bigint,
    N_POD_MANCANTI bigint,
    N_MISURE_1G bigint,
    N_MISURE_2G bigint,
    N_MISURE_MIX bigint,
    ENERGIA_1G_GIORNO double precision,
    ENERGIA_1G_BEST double precision,
    ENERGIA_2G_BEST double precision
)
partitioned by (SIMULAZIONE char(4), DATA date)
stored as parquet
location 's3://{NOME_BUCKET}/{tab_path}'
"""

client = boto3.client('redshift-data')

response = client.execute_statement(
    ClusterIdentifier='ewfqmevadr01',
    Database='dev',
    DbUser='evaqareddbo',
    Sql=create_string,
    WithEvent=False
)
while client.describe_statement(Id=response['Id'])['Status'] != 'FINISHED':
    time.sleep(5)
    if client.describe_statement(Id=response['Id'])['Status'] == 'FAILED':
        raise ConnectionError(client.describe_statement(Id=response['Id']))

add_partition_string = "alter table datamodel.eva_qa_glue_meter_consuntivo add if not exists\n"
data = LISTA_DATE[0][0]
data_fine = LISTA_DATE[-1][-1]
while data <= data_fine:
    add_partition_string += f"partition (SIMULAZIONE='{COD_SIMULAZIONE}', DATA='{data.strftime('%Y-%m-%d')}') location 's3://eva-qa-s3-model/datamodel/BEST_CHOICE/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data.strftime('%Y-%m-%d')}/'\n"
    data += relativedelta(days=1)

response = client.execute_statement(
    ClusterIdentifier='ewfqmevadr01',
    Database='dev',
    DbUser='evaqareddbo',
    Sql=add_partition_string,
    WithEvent=False
)
while client.describe_statement(Id=response['Id'])['Status'] != 'FINISHED':
    time.sleep(5)
    if client.describe_statement(Id=response['Id'])['Status'] == 'FAILED':
        raise ConnectionError(client.describe_statement(Id=response['Id']))

