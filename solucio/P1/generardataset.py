import os
import logging
import numpy as np
import csv

def generar_dataset(num, ind, dicc, fitxer_sortida):
	"""
	Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
	TODO: completar arguments, return. num és el número de files/ciclistes a generar. ind és l'index/identificador/dorsal.
	"""
	# TODO
	#Crea la carpeta si no existeix
	os.makedirs(os.path.dirname(fitxer_sortida), exist_ok=True)
	data = []
	dorsals_usats = set() #fem que els dorsals siguin únics (id)
	for tipus in dicc:
		for _ in range(num):
					while ind in dorsals_usats:
							ind += 1
					dorsals_usats.add(ind)

					#ES guarden les marques de cada participant
					temps_pujada = max(0, np.random.normal(tipus['mu_p'], tipus['sigma']))
					temps_baixada = max(0, np.random.normal(tipus['mu_b'], tipus['sigma']))
					temps_total = temps_pujada + temps_baixada

					#Guardem al dataset les els resultats
					data.append({
							"id": ind,
							"Tipus": tipus['name'],
							"tp": round(temps_pujada),
							"tb": round(temps_baixada),
							"tt": round(temps_total)
					})

	#ES genera el fitxer amb les dades dels ciclistes
	with open(fitxer_sortida, mode='w', newline='') as fout:
			writer = csv.DictWriter(fout, fieldnames=["id", "Tipus", "tp", "tb", "tt"])
			writer.writeheader()
			writer.writerows(data)

	return data

if __name__ == "__main__":

	str_ciclistes = 'data/ciclistes.csv'


	# BEBB: bons escaladors, bons baixadors
	# BEMB: bons escaladors, mal baixadors
	# MEBB: mal escaladors, bons baixadors
	# MEMB: mal escaladors, mal baixadors

	# Port del Cantó (18 Km de pujada, 18 Km de baixada)
	# pujar a 20 Km/h són 54 min = 3240 seg
	# pujar a 14 Km/h són 77 min = 4268 seg
	# baixar a 45 Km/h són 24 min = 1440 seg
	# baixar a 30 Km/h són 36 min = 2160 seg
	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]
	generar_dataset(1000, 1, dicc, str_ciclistes)
	logging.info("s'ha generat data/ciclistes.csv")