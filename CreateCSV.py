import json
import os

import pandas as pd


def load_path_json_files(path):
    files_in_directory = os.listdir(path)
    filtered_files = [
        file for file in files_in_directory if file.endswith(".json")]

    return filtered_files


def results_csv(paths, new_dir):
    df = pd.DataFrame({})
    for file in paths:
        json_file = open('{}/{}'.format('./Results/JSON/', file), "r")
        dictionary = json.load(json_file)
        json_file.close()

        df2 = pd.DataFrame(dictionary, index=[0])
        df = pd.concat([df, df2])

    if not os.path.exists('./Results/{}'.format(new_dir)):
        os.makedirs('./Results/{}'.format(new_dir))

    df.to_csv('./Results/{}/dataset.csv'.format(new_dir), index=False)


pathsJsons = load_path_json_files('./Results/JSON/')

# bisogna decidere il nome da dare alla nuova cartella
results_csv(pathsJsons, '30')
