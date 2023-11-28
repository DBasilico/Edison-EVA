import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])

from datetime import datetime
from pyspark.sql import SparkSession
import boto3

NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'

log_string = 'INIZIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

from lib_best_choice import *
log_string += 'LOAD LIBRERIA\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

now = datetime.now().date()

COD_SIMULAZIONE = sys.argv[1]
TRACE_ANALISI = eval(sys.argv[2])
INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()
EFFETTURARE_RECUPERO = eval(sys.argv[5])
GIORNI_INDIETRO_RECUPERO = int(sys.argv[6])
IS_PREVISIONE = eval(sys.argv[7])
DATA_BACKTEST = datetime.strptime(sys.argv[8], '%Y-%m-%d')

tab_path = f'datamodel/{NOME_TABELLA}/'

log_string += 'LOAD PARAMETRI\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

log_string += 'CARICO SESSIONE SPARK\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

##############################
### CONFIGURAZIONE CLUSTER ###
##############################

df_conf_cluster = get_conf_cluster(
    spark,
    nome_tabella_configurazione='CONFIGURAZIONE_CLUSTER',
    nome_bucket=NOME_BUCKET,
    cod_simulazione=COD_SIMULAZIONE
)

log_string += 'CONFIGURAZIONE CLUSTER\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

##################
### ANAGRAFICA ###
##################

if IS_PREVISIONE:
    df_anagrafica = get_anagrafica(
        spark,
        nome_tabella_anagrafica='output/',
        nome_bucket='eva-qa-s3-model',
        inizio_periodo=INIZIO_PERIODO,
        fine_periodo=INIZIO_PERIODO
    )
else:
    df_anagrafica = get_anagrafica(
        spark,
        nome_tabella_anagrafica='output/',
        nome_bucket='eva-qa-s3-model',
        inizio_periodo=INIZIO_PERIODO,
        fine_periodo=FINE_PERIODO
    )
log_string += 'ANAGRAFICA\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')
df_anagrafica = df_anagrafica.dropDuplicates(subset=['pod', 'DT_INI_VALIDITA', 'DT_FIN_VALIDITA'])

#df_anagrafica.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC01')

###############
### CLUSTER ###
###############

df_anagrafica = build_cluster(
    df_anagrafica=df_anagrafica,
    df_conf_cluster=df_conf_cluster
)
log_string += 'CLUSTER\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

#df_anagrafica.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC02')

######################
### CALENDARIO-POD ###
######################

df_best_choice = add_period(
    spark,
    nome_bucket=NOME_BUCKET,
    nome_tabella_calendario='CALENDARIO_GME',
    df_anagrafica=df_anagrafica,
    inizio_periodo=INIZIO_PERIODO,
    fine_periodo=FINE_PERIODO
)
log_string += 'CALENDARIO\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

#df_best_choice.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC03')

###########################
### AGGIUSTIAMO I CAMPI ###
###########################

df_best_choice = add_consumi(
    spark,
    nome_bucket=NOME_BUCKET,
    effettuare_recupero=EFFETTURARE_RECUPERO,
    inizio_periodo=INIZIO_PERIODO,
    fine_periodo=FINE_PERIODO,
    giorni_indietro_recupero=GIORNI_INDIETRO_RECUPERO,
    df_best_choice=df_best_choice,
    IS_PREVISIONE=IS_PREVISIONE,
    DATA_BACKTEST=DATA_BACKTEST
)
log_string += 'VALORI\n'
boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

#df_best_choice.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/BC04')

df_best_choice = df_best_choice.cache()

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

    log_string += 'TRACE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.sum('ENERGIA_NA_VALORE').alias('PREVISIONE_NA_RECUPERATI'),
        f.sum('ENERGIA_EAC_VALORE').alias('PREVISIONE_NA_EAC'),
        f.sum('CONSUMI').alias('PREVISIONE')
    )
    df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))
    df_best_choice = df_best_choice.withColumn('NOME_TRANSFORM', f.lit(NOME_TRANSFORM))
    df_best_choice = df_best_choice.withColumn('PRIMO_GIORNO_PREVISIONE', f.lit(INIZIO_PERIODO))

    data_corrente = INIZIO_PERIODO
    output_table = f's3://{NOME_BUCKET}/datamodel/FORECAST_SAGEMAKER'
    while data_corrente <= FINE_PERIODO:
        delete_folder_s3(bucket=NOME_BUCKET,
                         folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/NOME_TRANSFORM={NOME_TRANSFORM}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
        data_corrente += relativedelta(days=1)
    del data_corrente

    log_string += 'SVUOTAMENTO COMPLETATO\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'NOME_TRANSFORM', 'DATA']).parquet(output_table)

    log_string += 'SCRITTURA COMPLETATA\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

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

    log_string += 'TRACE\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

    df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
        f.count(f.lit(1)).cast('bigint').alias('N_POD'),
        f.sum('IS_MISSING').cast('bigint').alias('N_POD_MANCANTI'),
        f.sum('ENERGIA_1G_BEST_VALORE').alias('CONSUMI_1G_CERT'),
        f.sum('ENERGIA_1G_GIORNO_VALORE').alias('CONSUMI_1G_NOCERT'),
        f.sum('ENERGIA_2G_BEST_VALORE').alias('CONSUMI_2G_CERT'),
        f.sum('ENERGIA_NA_VALORE').alias('CONSUMI_NA_RECUPERATI'),
        f.sum('ENERGIA_EAC_VALORE').alias('CONSUMI_NA_EAC'),
        f.sum('CONSUMI').alias('CONSUMI'),
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

    log_string += 'SVUOTAMENTO COMPLETATO\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

    df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

    log_string += 'SCRITTURA COMPLETATA\n'
    boto3.client("s3").put_object(Body=log_string, Bucket=NOME_BUCKET, Key='log_best_choice.log')

if EFFETTURARE_RECUPERO:
    delete_recupero_na(nome_bucket=NOME_BUCKET, IS_PREVISIVO=IS_PREVISIONE, DATA_INIZIO=INIZIO_PERIODO)
