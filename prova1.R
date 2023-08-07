
library(stats)
library(cluster)
library(ggplot2)

setwd("/Users/mantioco/Documents/Progetto EVA/CSV")
anag<- read.csv("ANAG_SII.csv", sep=";", dec = ",",  
             stringsAsFactors=TRUE, na.strings=c("NA","", "NaN"))
head(anag)
str(anag)

print(names(anag))
#livelli <- levels(anag$id_area_gestionale)
#frequenze <- table(livelli)
#print(frequenze)

frequenze <- table(anag$id_area_gestionale)
freq1 <- table(anag$id_distributore)
# Stampa la tabella delle frequenze
print(frequenze)
print(freq1)

# Contare i valori mancanti per ciascuna colonna
valori_mancanti_per_colonna <- colSums(is.na(anag))
# Contare il numero totale di valori mancanti nel dataframe
totale_valori_mancanti <- sum(is.na(anag))
# Visualizzare i risultati
print(valori_mancanti_per_colonna)
print(totale_valori_mancanti)


#eliminare colonna id sistema origine ha 964185 valori mancanti 
anag$id_sistema_origine <- NULL

# Dividi il data frame in parti
res <- anag[anag$id_a %in% c("RES", "SORRENTO"),  ]
bus <- anag[anag$id_a %in% c("BUS-GRC", "BUS-MPA", "BUS-TOP"),  ]
enet <- anag[anag$id_a == "ENET", ]
ica <- anag[anag$id_a == "ICA", ]
gaxa <- anag[anag$id_a == "GAXA", ] # si potrebbe anche eliminare consumo annuo -1
amg <- anag[anag$id_a == "AMG-GAS", ]
pmi <- anag[anag$id_a == "PMI", ]
eng <- anag[anag$id_a == "ENG", ]

#provo ad eliminare tutte le righe con valori mancanti e sistemo il dataset
#anagnew <- na.omit(anag)
#anagnew$tms_updated <- NULL
#anagnew$dt_ini_dispacciamento <- NULL
#anagnew$dt_fin_dispacciamento <- NULL
#anagnew$tms_created <- NULL
#anagnew$dt_ini_validita <- NULL
#anagnew$dt_fin_validita <- NULL


str(anagnew)
livelli <- levels(anagnew$fornitura_prov)



#converto i valori in factor e in num
anagnew$id_societa_vendita <- as.factor(anagnew$id_societa_vendita )
anagnew$fornitura_cap <- as.factor(anagnew$fornitura_cap )
anagnew$consumo_annuo <- as.numeric(anagnew$consumo_annuo )
str(anagnew)


#library(clustMixType)
# clustering k-prototypes, che estende il classico algoritmo k-means per gestire
#sia le variabili numeriche che quelle categoriche


# Specifica l'indice delle variabili categoriche 
#indice_variabili_categoriche <- anagnew [, c(1:11, 13:17, 19)]

# Esegui il clustering k-prototypes
#k_prototypes_clustering <- clustMixType::kproto(anagnew, k = 3, clustermode = "hard", indcat = indice_variabili_categoriche)

# Visualizza i risultati del clustering
print(k_prototypes_clustering)

# Ottieni le etichette dei cluster assegnate a ciascuna osservazione
#etichette_cluster <- k_prototypes_clustering$cluster


# clustering k-prototypes, che estende il classico algoritmo k-means per gestire
#sia le variabili numeriche che quelle categoriche

# Specifica l'indice delle variabili categoriche 
indice_variabili_categoriche1 <- anag [, c(1:11, 13:17, 19)]

# Esegui il clustering k-prototypes
#k_prototypes_clustering <- clustMixType::kproto(anagnew, k = 3, clustermode = "hard", indcat = indice_variabili_categoriche)

# Visualizza i risultati del clustering
print(k_prototypes_clustering)

# Ottieni le etichette dei cluster assegnate a ciascuna osservazione
#etichette_cluster <- k_prototypes_clustering$cluster


