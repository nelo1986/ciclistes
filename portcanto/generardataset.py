"""
Modul generardataset.py
Aquest script genera dataset de ciclistes.
"""
import os
import logging
import csv
import numpy as np


def generar_dataset(num, ind, diccionari, fitxer_sortida):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
    completar arguments, return. num és el número de files/ciclistes a generar. ind 
    és l'index/identificador/dorsal.
    """
    #Crea la carpeta si no existe
    os.makedirs(os.path.dirname(fitxer_sortida), exist_ok=True)
    data = []
    dorsals_usats = set() #para que los dorsales sean únicos
    for tipus in diccionari:
        for _ in range(num):
            # Generar dorsal
            while ind in dorsals_usats:
                ind += 1
            dorsals_usats.add(ind)

            #Guardamos los tiempos
            temps_pujada = max(0, np.random.normal(tipus['mu_p'], tipus['sigma']))
            temps_baixada = max(0, np.random.normal(tipus['mu_b'], tipus['sigma']))
            temps_total = temps_pujada + temps_baixada

            # Los agregamos al dataset
            data.append({
                    "id": ind,
                    "Tipus": tipus['name'],
                    "tp": round(temps_pujada),
                    "tb": round(temps_baixada),
                    "tt": round(temps_total)
            })

    # Escriure dades al fitxer
    with open(fitxer_sortida, mode='w', newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id", "Tipus", "tp", "tb", "tt"])
        writer.writeheader()
        writer.writerows(data)

    return data

if __name__ == "__main__":

    STR_CICLISTES = 'data/ciclistes.csv'


    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240 # mitjana temps pujada bons escaladors
    MU_P_ME = 4268 # mitjana temps pujada mals escaladors
    MU_B_BB = 1440 # mitjana temps baixada bons baixadors
    MU_B_MB = 2160 # mitjana temps baixada mals baixadors
    SIGMA = 240 # 240 s = 4 min

    dicc = [
        {"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name":"MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]
    generar_dataset(1000, 1, dicc, STR_CICLISTES)
    logging.info("s'ha generat data/ciclistes.csv")
