""" @ IOC - Ruben Pabo - 2024 - CE IABD """
import logging
import pickle
import shutil
import mlflow
from mlflow.tracking import MlflowClient

from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

if __name__ == "__main__":

    logging.basicConfig(format='%(message)s', level=logging.INFO) # canviar entre DEBUG i INFO

    client = MlflowClient()
    experiment_name = "K sklearn ciclistes"
    exp = client.get_experiment_by_name(experiment_name)

    if not exp:
        print(f"Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name,
        tags={"mlflow.note.content": "portcanto variació de paràmetre K"})
        exp = client.get_experiment_by_name(experiment_name)

    mlflow.set_experiment("K sklearn portcanto")

    def get_run_dir(artifacts_uri):
        """ retorna ruta del run """
        return artifacts_uri[7:-10]

    def remove_run_dir(run_dir):
        """ elimina path amb shutil.rmtree """
        shutil.rmtree(run_dir, ignore_errors=True)

    runs = MlflowClient().search_runs(
        experiment_ids=[exp.experiment_id],
    )

    # esborrem tots els runs de l'experiment
    for run in runs:
        mlflow.delete_run(run.info.run_id)
        remove_run_dir(get_run_dir(run.info.artifact_uri))

    path_dataset = './data/ciclistes.csv'
    ciclistes_data = load_dataset(path_dataset)
    ciclistes_data = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data)
    Ks = [2, 3, 4, 5, 6, 7, 8]

    for K in Ks:
        #dataset = mlflow.data.from_pandas(ciclistes_data, source=path_dataset)

        mlflow.start_run(description='K={}'.format(K))
        #mlflow.log_input(dataset, context='training')

        clustering_model = clustering_kmeans(ciclistes_data, K)
        data_labels = clustering_model.labels_
        h_score = round(homogeneity_score(true_labels, data_labels), 5)
        c_score = round(completeness_score(true_labels, data_labels), 5)
        v_score = round(v_measure_score(true_labels, data_labels), 5)
        logging.info('K: %d', K)
        logging.info('H-measure: %.5f', h_score)
        logging.info('C-measure: %.5f', c_score)
        logging.info('V-measure: %.5f', v_score)

        tags = {
            "engineering": "RPA-IOC",
            "release.candidate": "RC1",
            "release.version": "1.1.2",
        }
        mlflow.set_tags(tags)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        mlflow.log_param("K", K)

        mlflow.log_metric("h", h_score)
        mlflow.log_metric("c", c_score)
        mlflow.log_metric("v_score", v_score)

        mlflow.log_artifact("./data/ciclistes.csv")
        mlflow.end_run()
