import logging
import pickle
from copy import deepcopy
from datetime import datetime
from os.path import join

import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import build_all_campus_dataframe, gather_f1_measures

if __name__ == '__main__':
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=fr'logs\run_ml_models_{date_str}.log', filemode='w')

    models_files = [('best_nb.pkl', False), ('best_tree.pkl', False), ('best_mlp.pkl', True), ('best_svm.pkl', True)]

    # unpickle the models
    models = []
    for model_file, needs_scaler in models_files:
        name = model_file[:-4]
        model_file = join('models', model_file)
        with open(model_file, 'rb') as f:
            models.append((pickle.load(f), needs_scaler, name))

    all_train_data, all_test_data = build_all_campus_dataframe()
    x_train, y_train = all_train_data.drop('FAIL', axis=1), all_train_data['FAIL']
    x_test, y_test = all_test_data.drop('FAIL', axis=1), all_test_data['FAIL']

    scaler = StandardScaler()
    temp_data = deepcopy(x_train)
    X_train_scaled = scaler.fit_transform(temp_data.drop(columns=['user_id', 'course_id', 'user_state']))

    for model, needs_scaler, name in models:
        print(name)
        pass_scaler = scaler if needs_scaler else None
        pred_data = deepcopy(all_test_data)
        results = gather_f1_measures(model, pred_data, scaler=pass_scaler)
        file_out = join('results', f'{name}_results.csv')
        pd.DataFrame(results).to_csv(file_out, index=False)