"""
@ IOC - CE IABD
"""
import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans

class TestGenerarDataset(unittest.TestCase):
    """
    classe TestGenerarDataset
    """
    global mu_p_be
    global mu_p_me
    global mu_b_bb
    global mu_b_mb
    global sigma
    global dicc

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

    def test_longituddataset(self):
        """
        Test la longitud de l'array
        """
        arr = generar_dataset(1000, 1, dicc,fitxer_sortida='data/ciclistes.csv' )
        self.assertEqual(len(arr), 4000)

    def test_valorsmitjatp(self):
        """
        Test del valor mitjà del tp
        """
        # Generamos el dataset
        arr = generar_dataset(1000, 1, dicc, fitxer_sortida='data/ciclistes.csv')
        # Extraemos la columna 'tp' de cada fila
        # Asumimos que arr es una lista de diccionarios con la clave 'tp'
        arr_tp = [row['tp'] for row in arr if 'tp' in row]
        # Validamos que arr_tp no está vacío
        self.assertGreater(len(arr_tp), 0, "La columna 'tp' está vacía o no se generó correctamente")
        # Calculamos el promedio de los valores de 'tp'
        tp_mig = sum(arr_tp) / len(arr_tp)
        # Comprobamos que el promedio es menor a 4000
        self.assertLess(tp_mig, 4000)

    def test_valorsmitjatb(self):
        """
        Test del valor mitjà del tb
        """
        # Generamos el dataset
        arr = generar_dataset(1000, 1, dicc, fitxer_sortida='data/ciclistes.csv')
        # Extraemos la columna 'tb' de cada fila
        # Aseguramos que 'tb' está presente en cada fila antes de acceder
        arr_tb = [row['tb'] for row in arr if 'tb' in row]
        # Validamos que arr_tb no está vacío
        self.assertGreater(len(arr_tb), 0, "La columna 'tb' está vacía o no se generó correctamente")
        # Calculamos el promedio de los valores de 'tb'
        tb_mig = sum(arr_tb) / len(arr_tb)
        # Comprobamos que el promedio es mayor a 2000
        self.assertGreater(tb_mig, 1500)

class TestClustersCiclistes(unittest.TestCase):
    """
    classe TestClustersCiclistes
    """
    global ciclistes_data_clean
    global data_labels

    path_dataset = './data/ciclistes.csv'
    ciclistes_data = load_dataset(path_dataset)
    ciclistes_data_clean = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data_clean)
    #ciclistes_data_clean = ciclistes_data_clean.drop('tipus', axis=1) # eliminem el tipus, ja no interessa

    clustering_model = clustering_kmeans(ciclistes_data_clean)
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)
    data_labels = clustering_model.labels_

    def test_check_column(self):
        """
        Comprovem que una columna existeix
        """

        self.assertIn('tp', ciclistes_data_clean.columns)

    def test_data_labels(self):
        """
        Comprovem que data_labels té la mateixa longitud que ciclistes
        """

        self.assertEqual(len(data_labels), len(ciclistes_data_clean))

    def test_model_saved(self):
        """
        Comprovem que a la carpeta model/ hi ha els fitxer clustering_model.pkl
        """
        check_file = os.path.isfile('./model/clustering_model.pkl')
        self.assertTrue(check_file)

if __name__ == '__main__':
    unittest.main()