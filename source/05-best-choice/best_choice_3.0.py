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

COD_SIMULAZIONE = sys.argv[1]
NOME_TABELLA = 'BEST_CHOICE'
NOME_BUCKET = 'eva-qa-s3-model'
TRACE_ANALISI = eval(sys.argv[2])

INIZIO_PERIODO = datetime.strptime(sys.argv[3], '%Y-%m-%d').date()
FINE_PERIODO = datetime.strptime(sys.argv[4], '%Y-%m-%d').date()

EFFETTURARE_RECUPERO = eval(sys.argv[5])

tab_path = f'datamodel/{NOME_TABELLA}/'

now = datetime.now()

spark = SparkSession.builder.appName('EVA - BEST CHOICE').enableHiveSupport().getOrCreate()

#############################
### GESTIONE DEI RECUPERI ###
#############################

if EFFETTURARE_RECUPERO:
    GIORNI_INDIETRO_RECUPERO = int(sys.argv[6])
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
df_anagrafica = df_anagrafica.withColumn('DT_INI_VALIDITA', f.to_timestamp(f.col('DT_INI_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_anagrafica = df_anagrafica.withColumn('DT_FIN_VALIDITA', f.to_timestamp(f.col('DT_FIN_VALIDITA'), 'dd/MM/yyyy HH:mm:SS'))
df_anagrafica = df_anagrafica.withColumn('CONSUMO_ANNUO_COMPLESSIVO', f.regexp_replace(f.col('CONSUMO_ANNUO_COMPLESSIVO'), ',', '.').cast(DoubleType()))

# Filtriamo solo i pod che hanno inizio e fine fornitura nella data corrente
df_anagrafica = df_anagrafica.filter(
    (f.col('DT_FIN_VALIDITA') >= now) &
    (f.col('DT_INI_VALIDITA') <= now)
)

# CODIZIONI PER CAPIRE SE POD E' 1G OPPURE 2G
df_anagrafica = df_anagrafica.withColumn('FLUSSO_DATI', f.when(
    f.col('TIPO_MISURATORE') == 'G',
    f.lit('2G')
).otherwise(
    f.lit('1G')
))

# TODO: gestione degli errori in anagrafica

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
                                    f.coalesce(f.col('PROVINCIA_CRM'), f.lit('')),
                                    f.coalesce(f.col('FLUSSO_DATI'), f.lit('')),
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
            df_anagrafica = df_anagrafica.filter(regola_where)
        df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.expr(row['REGOLA_SELECT']))

df_anagrafica = df_anagrafica.withColumn('CLUSTER', f.when(
    ((f.col('CLUSTER').isNull()) & (f.col('FLUSSO_DATI') == '1G')), f.lit('CLUSTER_MANCANTI_1G')
).when(
    ((f.col('CLUSTER').isNull()) & (f.col('FLUSSO_DATI') == '2G')), f.lit('CLUSTER_MANCANTI_2G')
).when(
    ((f.col('CLUSTER').isNull()) & (~(f.col('FLUSSO_DATI').isin(['1G', '2G'])))), f.lit('CLUSTER_MANCANTI')
).otherwise(
    f.col('CLUSTER')
))

df_anagrafica = df_anagrafica.select(
    f.col('POD'),
    f.col('CLUSTER'),
    f.col('FLUSSO_DATI'),
    f.col('CONSUMO_ANNUO_COMPLESSIVO').alias('EAC'),
    f.col('ZONA'),
    f.col('ID_AREA_GESTIONALE'),
    f.col('PROVINCIA_CRM')
)

# TODO: eliminare drop duplicates: non dovrebbero esistere pod doppi
df_anagrafica = df_anagrafica.dropDuplicates(subset=['POD'])

######################
### CALENDARIO-POD ###
######################

df_calendario = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
    .filter(f.col('DATA').between(INIZIO_PERIODO, FINE_PERIODO)) \
    .select(
        'UNIX_TIME',
        'TIMESTAMP',
        'DATA',
        'ORA_GME',
        'GIORNO_SETTIMANA'
)

df_num_giorni_anno = spark.read.parquet(f's3://{NOME_BUCKET}/datamodel/CALENDARIO_GME') \
    .groupBy('ANNO').count().withColumnRenamed('count', 'NUM_GIORNO_ORA')

df_best_choice = df_anagrafica.crossJoin(df_calendario)
df_best_choice = df_best_choice.withColumn('ANNO', f.year(f.col('DATA')))
df_best_choice = df_best_choice.join(df_num_giorni_anno, on=['ANNO'], how='left')
df_best_choice = df_best_choice.withColumn('EAC_GIORNO_ORA', f.col('EAC') / f.col('NUM_GIORNO_ORA')) \
    .drop('NUM_GIORNO_ORA', 'ANNO')

# Windows function per recupero dati mancanti
w_recupero_na = Window.partitionBy('POD', 'ORA_GME').orderBy(f.col('DATA').desc()).rowsBetween(Window.currentRow, Window.unboundedFollowing)

##########################
### MISURE 1G - GIORNO ###
##########################

df_1g_giorno = extract_1g_giorno(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_1g_giorno, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')
del df_1g_giorno

########################
### MISURE 1G - BEST ###
########################

df_1g_best = extract_1g_best(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_1g_best, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')
del df_1g_best

########################
### MISURE 2G - BEST ###
########################

df_2g_best = extract_2g_best(spark, INIZIO_PERIODO, FINE_PERIODO)
df_best_choice = df_best_choice.join(df_2g_best, on=['POD', 'DATA', 'ORA_GME', 'FLUSSO_DATI'], how='left')
del df_2g_best

###########################
### AGGIUSTIAMO I CAMPI ###
###########################

if df_recupero_na is not None:
    df_best_choice = df_best_choice.join(df_recupero_na, on=['POD', 'ORA_GME', 'GIORNO_SETTIMANA'], how='left')
else:
    df_best_choice = df_best_choice.withColumn('ENERGIA_1G_RECUPERO_NA', f.lit(None).cast(DoubleType()))
    df_best_choice = df_best_choice.withColumn('ENERGIA_2G_RECUPERO_NA', f.lit(None).cast(DoubleType()))

df_best_choice = df_best_choice.withColumn('ENERGIA_1G_BEST_VALORE', f.when(
    f.col('ENERGIA_1G_BEST').isNotNull(), f.col('ENERGIA_1G_BEST')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_1G_GIORNO_VALORE', f.when(
    (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNotNull()),
    f.col('ENERGIA_1G_GIORNO')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_2G_BEST_VALORE', f.when(
    f.col('ENERGIA_2G_BEST').isNotNull(),
    f.col('ENERGIA_2G_BEST')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_NA_VALORE', f.when(
    ((f.col('FLUSSO_DATI') == '1G') & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull())),
    f.col('ENERGIA_1G_RECUPERO_NA')
).when(
    ((f.col('FLUSSO_DATI') == '2G') & (f.col('ENERGIA_2G_BEST').isNull())),
    f.col('ENERGIA_1G_RECUPERO_NA')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('ENERGIA_EAC_VALORE', f.when(
    ((f.col('FLUSSO_DATI') == '1G') & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull()) & (f.col('ENERGIA_NA_VALORE').isNull())) |
    ((f.col('FLUSSO_DATI') == '2G') & (f.col('ENERGIA_2G_BEST').isNull()) & (f.col('ENERGIA_NA_VALORE').isNull())),
    f.col('EAC_GIORNO_ORA')
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('CONSUMI', f.when(
    f.col('FLUSSO_DATI') == '1G',
    f.coalesce(f.col('ENERGIA_1G_BEST'), f.col('ENERGIA_1G_GIORNO'), f.col('ENERGIA_1G_RECUPERO_NA'), f.col('EAC_GIORNO_ORA'))
).when(
    f.col('FLUSSO_DATI') == '2G',
    f.coalesce(f.col('ENERGIA_2G_BEST'), f.col('ENERGIA_2G_RECUPERO_NA'), f.col('EAC_GIORNO_ORA'))
).otherwise(
    f.lit(None)
))

df_best_choice = df_best_choice.withColumn('FLAG_DATO_MANCANTE', f.when(
    ((f.col('FLUSSO_DATI') == '1G') & (f.col('ENERGIA_1G_BEST').isNull()) & (f.col('ENERGIA_1G_GIORNO').isNull())) |
    ((f.col('FLUSSO_DATI') == '2G') & (f.col('ENERGIA_2G_BEST').isNull())),
    f.lit(1)
).otherwise(
    f.lit(0)
))

#############################################
### SCRITTURA TABELLA ANALISI BEST-CHOICE ###
#############################################

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

df_best_choice = df_best_choice.groupBy('CLUSTER', 'DATA', 'UNIX_TIME', 'ORA_GME', 'TIMESTAMP', 'FLUSSO_DATI').agg(
    f.count(f.lit(1)).cast('bigint').alias('N_POD'),
    f.sum('FLAG_DATO_MANCANTE').cast('bigint').alias('N_POD_MANCANTI'),
    f.sum('ENERGIA_1G_BEST_VALORE').alias('CONSUMI_1G_BEST'),
    f.sum('ENERGIA_1G_GIORNO_VALORE').alias('CONSUMI_1G_GIORNO'),
    f.sum('ENERGIA_2G_BEST_VALORE').alias('CONSUMI_2G_BEST'),
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

df_best_choice.repartition(50, 'CLUSTER').write.mode('append').partitionBy(['SIMULAZIONE', 'DATA']).parquet(output_table)

if df_recupero_na is not None:
   delete_folder_s3(bucket=NOME_BUCKET, folder_path=df_recupero_na)