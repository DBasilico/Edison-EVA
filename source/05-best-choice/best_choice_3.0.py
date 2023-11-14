import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
import pandas as pd
from dateutil.relativedelta import relativedelta
from lib_best_choice import extract_1g_giorno, extract_1g_best, extract_2g_best, build_recupero_na, delete_folder_s3

NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'

COD_SIMULAZIONE = sys.argv[1]
TRACE_ANALISI = eval(sys.argv[2])
INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()
EFFETTURARE_RECUPERO = eval(sys.argv[5])
GIORNI_INDIETRO_RECUPERO = int(sys.argv[6])
EFFETTUARE_PREVISIONE = eval(sys.argv[7])
LAG_PREVISIONE = int(sys.argv[8]) if EFFETTUARE_PREVISIONE else 0

tab_path = f'datamodel/{NOME_TABELLA}/'

now = datetime.now()

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

#############################
### GESTIONE DEI RECUPERI ###
#############################

if EFFETTURARE_RECUPERO:
    INIZIO_PERIODO_RECUPERO = INIZIO_PERIODO - relativedelta(days=1+GIORNI_INDIETRO_RECUPERO)
    FINE_PERIODO_RECUPERO = INIZIO_PERIODO - relativedelta(days=1)
    df_recupero_na = spark.read.parquet(
        build_recupero_na(spark, INIZIO_PERIODO_RECUPERO, FINE_PERIODO_RECUPERO, NOME_BUCKET)
    )
else:
    df_recupero_na = None

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

##################
### ANAGRAFICA ###
##################

# POD attivi in cui dobbiamo calcolare la BEST CHOICE (DATA FORNITURA)
df_anagrafica = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII')
df_anagrafica = df_anagrafica.filter(f.col('TRATTAMENTO') == 'O')
df_anagrafica = df_anagrafica.withColumn('TIPO_FLUSSO', f.when(f.col('TIPO_MISURATORE') == 'G', f.lit('2G')).otherwise(f.lit('1G')))
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))

# Filtriamo solo i pod che hanno inizio e fine fornitura nella data corrente
df_anagrafica = df_anagrafica.filter(
    (f.col('DT_FIN_VALIDITA') >= now) &
    (f.col('DT_INI_VALIDITA') <= now)
)

# TODO: gestione degli errori in anagrafica

##########################
### ANAGRAFICA PER EAC ###
##########################

df_eac = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/ANAG_SII')
df_eac = df_eac.filter(f.col('TRATTAMENTO') == 'O')
df_eac = df_eac.withColumn('DT_INI_VALIDITA', f.to_date(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_eac = df_eac.withColumn('DT_FIN_VALIDITA', f.to_date(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_eac = df_eac.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))

df_eac = df_eac.groupBy('POD', 'DT_INI_VALIDITA', 'DT_FIN_VALIDITA').agg(f.avg('CONSUMO_ANNUO_COMPLESSIVO').alias('EAC'))
df_eac = df_eac.withColumn('DATA', f.expr('sequence(DT_INI_VALIDITA, DT_FIN_VALIDITA, interval 1 day)'))
df_eac = df_eac.withColumn('NUMERO_GIORNI_PERIODO', f.size(f.col('DATA')))
df_eac = df_eac.withColumn('DATA', f.explode('DATA'))
df_eac = df_eac.withColumn('EAC', f.col('EAC')/f.col('NUMERO_GIORNI_PERIODO')).drop('NUMERO_GIORNI_PERIODO')

df_num_ore_giorno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME').groupBy('DATA').count().withColumnRenamed('count', 'NUM_ORE_GIORNO')

df_eac = df_eac.join(df_num_ore_giorno, on=['DATA'], how='left')

df_eac = df_eac.withColumn('EAC', f.col('EAC')/f.col('NUM_ORE_GIORNO')).drop('NUM_ORE_GIORNO')

df_eac = df_eac.groupBy('POD', 'DATA').agg(f.avg('EAC').alias('EAC'))

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
        concat_syntax = f.concat_ws('#',
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
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(f'CASE WHEN {regola_where} THEN {regola_select} ELSE CLUSTER END'))
        else:
            df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(regola_select))

df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.coalesce(f.col('CLUSTER'), f.lit('CLUSTER_MANCANTI')))

df_anagrafica.repartition(1).write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/ANAGRAFICA_CLUSTER')

df_anagrafica = df_anagrafica.select(
    f.col('POD'),
    f.col('CLUSTER')
)

# TODO: eliminare drop duplicates: non dovrebbero esistere pod doppi
df_anagrafica = df_anagrafica.dropDuplicates(subset=['POD'])

######################
### CALENDARIO-POD ###
######################

df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
    .filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO + relativedelta(days=LAG_PREVISIONE))) \
    .select(
        'UNIX_TIME',
        'TIMESTAMP',
        'DATA',
        'ORA_GME',
        'GIORNO_SETTIMANA'
)

OFFSET = df_calendario.filter(f.col('DATA').between(FINE_PERIODO+relativedelta(days=1), FINE_PERIODO+relativedelta(days=LAG_PREVISIONE))).count()

df_best_choice = df_anagrafica.crossJoin(df_calendario)

df_best_choice = df_best_choice.join(df_eac, on=['POD', 'DATA'], how='left')

##########################
### MISURE 1G - GIORNO ###
##########################

df_1g_giorno = extract_1g_giorno(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_1g_giorno, on=['POD', 'DATA', 'ORA_GME'], how='left')
df_1g_giorno.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/1G_GIORNO')
del df_1g_giorno

########################
### MISURE 1G - BEST ###
########################

df_1g_best = extract_1g_best(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
df_1g_best.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/1G_BEST')
del df_1g_best

########################
### MISURE 2G - BEST ###
########################

df_2g_best = extract_2g_best(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'DATA', 'ORA_GME'], how='left')
df_2g_best.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/2G_BEST')
del df_2g_best

###########################
### AGGIUSTIAMO I CAMPI ###
###########################

if df_recupero_na is not None:
    df_best_choice = df_best_choice.join(df_recupero_na, on=['POD', 'ORA_GME', 'GIORNO_SETTIMANA'], how='left')
else:
    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_RECUPERO_NA', f.lit(None).cast(DoubleType()))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_RECUPERO_NA', f.lit(None).cast(DoubleType()))

df_best_choice = df_best_choice.withColumn('IS_MISSING', f.when(
    (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_2G_BEST').isNull()),
    f.lit(1)
).otherwise(
    f.lit(0)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_NA_VALORE', f.when(
    f.col('IS_MISSING') == 1,
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
    f.col('ENERGIA_1G_RECUPERO_NA'),
    f.col('ENERGIA_2G_RECUPERO_NA'),
    f.col('EAC')
))

window_lag = Window.partitionBy('POD').orderBy(f.col('UNIX_TIME').asc())

df_best_choice = df_best_choice.withColumn('PREVISIONE', f.lag(f.col('CONSUMI'), offset=OFFSET).over(window_lag))

#############################################
### SCRITTURA TABELLA ANALISI BEST-CHOICE ###
#############################################

df_best_choice.write.mode('overwrite').parquet(f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}_ANALISI')

if TRACE_ANALISI:
    outpath_trace = f's3://{NOME_BUCKET}/datamodel/{NOME_TABELLA}_ANALISI'
    data_corrente = INIZIO_PERIODO
    while data_corrente <= FINE_PERIODO:
        delete_folder_s3(bucket=NOME_BUCKET,
                         folder_path=f'datamodel/{NOME_TABELLA}_ANALISI/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
        data_corrente += relativedelta(days=1)
    del data_corrente
    df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE)).write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(outpath_trace)

############################
### AGGREGAZIONE CLUSTER ###
############################

if EFFETTUARE_PREVISIONE:
    df_best_choice_previsione = df_best_choice.filter(f.col('DATA').between(FINE_PERIODO+relativedelta(days=1), FINE_PERIODO+relativedelta(days=LAG_PREVISIONE)))
    df_best_choice = df_best_choice.filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO))

    df_best_choice_previsione = df_best_choice_previsione.groupBy('CLUSTER', 'UNIX_TIME').agg(
        f.sum('PREVISIONE').alias('PREVISIONE')
    )
    df_best_choice_previsione = df_best_choice_previsione.withColumn('NOME_TRANSFORM', f.lit('Benchmark'))
    df_best_choice_previsione = df_best_choice_previsione.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

    df_best_choice_previsione.write.mode('overwrite').parquet('s3://eva-qa-s3-model/datamodel/FORECAST_SAGEMAKER/BENCHMARK')

df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP').agg(
    f.count(f.lit(1)).cast('bigint').alias('N_POD'),
    f.sum('IS_MISSING').cast('bigint').alias('N_POD_MANCANTI'),
    f.sum('ENERGIA_1G_BEST').alias('CONSUMI_1G_BEST'),
    f.sum('ENERGIA_1G_GIORNO').alias('CONSUMI_1G_GIORNO'),
    f.sum('ENERGIA_2G_BEST').alias('CONSUMI_2G_BEST'),
    f.sum('ENERGIA_NA_VALORE').alias('CONSUMI_NA_RECUPERATI'),
    f.sum('ENERGIA_EAC_VALORE').alias('CONSUMI_NA_EAC'),
    f.sum('CONSUMI').alias('CONSUMI')
)

df_best_choice = df_best_choice.withColumn('SIMULAZIONE', f.lit(COD_SIMULAZIONE))

data_corrente = INIZIO_PERIODO
output_table = f's3://{NOME_BUCKET}/{tab_path}'
while data_corrente <= FINE_PERIODO:
    delete_folder_s3(bucket=NOME_BUCKET,
                     folder_path=f'datamodel/{NOME_TABELLA}/SIMULAZIONE={COD_SIMULAZIONE}/DATA={data_corrente.strftime("%Y-%m-%d")}/')
    data_corrente += relativedelta(days=1)
del data_corrente

df_best_choice.write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

if df_recupero_na is not None:
    delete_folder_s3(bucket=NOME_BUCKET, folder_path=df_recupero_na)
