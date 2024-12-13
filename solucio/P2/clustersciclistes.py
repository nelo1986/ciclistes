"""
@ IOC - CE IABD
"""
import os
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w', encoding="utf-8") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

#
def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes
    arguments:
        path -- dataset
    Returns: dataframe
    """
    return pd.read_csv(path, delimiter=',')

def eda(df):
    """
    Exploratory Data Analysis del dataframe
    arguments:
        df -- dataframe
    Returns: None
    """
    logging.debug('\n%s', df.shape)
    logging.debug('\n%s', df[:5])
    logging.debug('\n%s', df.columns)
    logging.debug('\n%s', df.info())

def clean(df):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers
    arguments:
        df -- dataframe
    Returns: dataframe
    """
    # eliminem les columnes que no interessen
    df = df.drop('id', axis=1)
    df = df.drop('tt', axis=1)
    logging.debug('\nDataframe:\n%s\n...', df[:3])
    return df

def extract_true_labels(df):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)
    arguments:
        df -- dataframe
    Returns: numpy ndarray (true labels)
    """
    true_labels = df['Tipus'].to_numpy()
    df.drop('Tipus', axis=1, inplace=True)#Elimina col tipus
    logging.debug('\nEtiquetes úniques: %s\n', true_labels)
    return true_labels

def visualitzar_pairplot(df):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.
    arguments:
        df -- dataframe
    Returns: None
    """
    sns.pairplot(df)
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass
    plt.savefig("img/pairplot.png")

def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model
    arguments:
        data -- les dades: tp i tb
    Returns: model (objecte KMeans)
    """
    model = KMeans(n_clusters, random_state=42)
    with suppress_stdout_stderr():
        model.fit(data)
    return model

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs
    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades 
        a un dels 4 clústers)
    Returns: None
    """
    try:
        os.makedirs(os.path.dirname('img/'))
    except FileExistsError:
        pass
    fig = plt.figure()
    sns.scatterplot(x='tb', y='tp', data=data, hue=labels, palette="rainbow")
    plt.savefig("img/grafica1.png")
    fig.clf()
    #plt.show()

def associar_clusters_patrons(tipus, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.
    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat
    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    dicc = {'tp':0, 'tb': 1}
    logging.info('Centres:')
    for j in range(len(tipus)):
        logging.info(
            '{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(
                j, model.cluster_centers_[j][dicc['tp']],model.cluster_centers_[j][dicc['tb']]))

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(clustering_model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus

def generar_informes(df, tipus):
    """
    Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers,
    i generem 4 fitxers de ciclistes per cadascun dels clústers
    arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers
    Returns: None
    """
    ciclistes_label = [
        df[df['label'] == 0],
        df[df['label'] == 1],
        df[df['label'] == 2],
        df[df['label'] == 3]
    ]
    try:
        os.makedirs(os.path.dirname('informes/'))
    except FileExistsError:
        pass

    for tip in tipus:
        fitxer = tip['name'].replace(' ', '_') + '.txt'
        foutput = open("informes/" + fitxer, "w")
        t = [t for t in tipus if t['name'] == tip['name']]
        ciclistes = ciclistes_label[t[0]['label']].index
        for ciclista in ciclistes:
            foutput.write(str(ciclista) + '\n')
        foutput.close()
    logging.info('\nS\'han generat els informes en la carpeta informes/')

def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers
    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    df_dades_ciclista = pd.DataFrame(columns=['id', 'tp', 'tb', 'tt'], data=dades)
    df_dades_ciclista = clean(df_dades_ciclista)
    return df_dades_ciclista, model.predict(df_dades_ciclista)
# ----------------------------------------------

if __name__ == "__main__":

    PATH_DATASET = './data/ciclistes.csv'
    """
    load_dataset
    EDA
    clean
    extract_true_labels
    eliminem el tipus, ja no interessa .drop('tipus', axis=1)
    visualitzar_pairplot
    clustering_kmeans
    pickle.dump(...) guardar el model
    mostrar scores i guardar scores
    visualitzar_clusters
    afegim la columna label al dataframe
    associar_clusters_patrons(tipus, clustering_model)
    guardem la variable tipus a model/tipus_dict.pkl
    generar_informes
    """
    logging.basicConfig(format='%(message)s', level=logging.DEBUG) # canviar entre DEBUG i INFO
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    ciclistes_data = load_dataset(PATH_DATASET)
    eda(ciclistes_data)

    ciclistes_data = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data)
    visualitzar_pairplot(ciclistes_data)
    clustering_model = clustering_kmeans(ciclistes_data)

      # guardem el model
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    data_labels = clustering_model.labels_
    logging.info('\nHomogeneity: %.3f', homogeneity_score(true_labels, data_labels))
    logging.info('Completeness: %.3f', completeness_score(true_labels, data_labels))
    logging.info('V-measure: %.3f', v_measure_score(true_labels, data_labels))
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump({
            "h": homogeneity_score(true_labels, data_labels),
            "c": completeness_score(true_labels, data_labels),
            "v": v_measure_score(true_labels, data_labels)}, f)
    visualitzar_clusters(ciclistes_data, data_labels)
    ciclistes_data['label'] = clustering_model.labels_.tolist()
    logging.debug('\nColumna label:\n%s', ciclistes_data[:5])
    #array de diccionaris que assignarà els tipus als labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
    tipus = associar_clusters_patrons(tipus, clustering_model)

    generar_informes(ciclistes_data, tipus)

    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550] # MEMB
    ]

    logging.debug('\nNous valors:\n%s', nous_ciclistes)
    df_ciclistes_data, pred = nova_prediccio(nous_ciclistes, clustering_model)
    logging.info('\nPredicció dels valors:\n%s', pred)
    #Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('tipus %s (%s) - classe %s', ciclistes_data.index[i], t[0]['name'], p)
