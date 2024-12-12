# Table of contents
1. [Introducció al projecte](#projecte)
2. [Execució](#run)
3. [Testing](#tests)
4. [Llicència](#licence)

# Introducció al projecte <a name="projecte"></a>
**portcanto** El Port del Cantó és un port de muntanya que uneix les comarques de l’Alt Urgell (Adrall) i el Pallars Sobirà (Sort). Són 18Km de pujada i 18Km de baixada, que típicament es puja entre 54 i 77min; i es baixa entre 24 i 36min. Es generaran dades sintètiques que simularan una cursa ciclista entre Adrall i Sort.

**Objectiu:** Descobrir els 4 patrons amb l'algoritme de clustering KMeans.

**Components del projecte:**
- **generardataset**: Crea dades sintètiques dels ciclistes i les guarda a la carpeta `data/`.
- **clusterciclistes**: Analitza les dades:
  - Càrrega, neteja i anàlisi del dataset (elimina columnes `id` i `tt`).
  - Extracció de les etiquetes (`Tipus`), que s’eliminen del dataframe.
  - Visualitza qualitativament els clústers amb `visualitzar_pairplot`.
  - Aplica KMeans, guarda el model a `model/clustering_model.pkl` i les mètriques d’homogeneïtat, completesa i V-measure a `model/scores.pkl`.
  - Genera informes per cada clúster (`BEBB.txt`, etc.).
  - Classifica nous valors en els clústers corresponents.

# Execució <a name="run"></a>

Pots crear un entorn virtual fent:
```
$ python -m venv venv
 o bé:
$ virtualenv venv

$ source venv/bin/activate
```

i tot seguit instal·lar els mòduls necessaris:
```
$ pip install -r requirements.txt
```

Per executar:
```
$ python generardataset.py
```

```
$ python clustersciclistes.py [cli | gui]
```

# Testing <a name="tests"></a>

Des de l'arrel del projecte:
```
$ python -m unittest discover -s tests
```

# Llicència <a name="licence"></a>
Rubén Pabó Amores - IOC (2024)
Llicència MIT. [LICENSE.txt](LICENSE.txt) per més detalls


