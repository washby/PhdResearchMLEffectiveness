import json
import logging
import pickle

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from settings import DB_CONFIG_FILENAME
import database


def create_pickle_file(table_name, output_filename, db_config_filename=DB_CONFIG_FILENAME):
    """
    Creates a pickle file from a table in the database
    :param table_name:
    :param output_filename:
    :param db_config_filename:
    :return:
    """
    db = database.DatabaseUtility(db_config_filename)
    logging.info(f'Connecting to database for {output_filename}')
    db.establish_connection()
    logging.info('Connected to database')
    df = db.select_all_from_table(table_name)
    logging.info(f"for {output_filename} the df shape is {len(df)}")
    pickled_df = pickle.dumps(df)
    db.close_connection()
    with(open(output_filename, 'wb')) as f:
        f.write(pickled_df)


def build_test_and_train_data(df, term_id_list):
    logging.info("Building test and train data")
    logging.info(f"df shape: {df.shape}")
    train_data = df[~df['term_id'].isin(term_id_list)]
    test_data = df[df['term_id'].isin(term_id_list)]
    return train_data, test_data

def gather_f1_measures(model, test_data, scaler=None):
    results = {'course_id': [], 'f1_measure': [], 'student_cnt': [], 'accuracy': [],
               'tn': [], 'fp': [], 'fn': [], 'tp': []}
    grouped_by_crs = test_data.groupby('course_id')
    # confusion = None
    for crs_id, crs_df in grouped_by_crs:
        outcome = crs_df['FAIL']
        input_data = crs_df.drop(columns=['user_id', 'course_id', 'user_state', 'FAIL'])
        if scaler:
            input_data = scaler.fit_transform(input_data)
        pred = model.predict(input_data)

        accuracy = accuracy_score(outcome, pred)
        confusion = confusion_matrix(outcome, pred)
        if len(confusion) == 1:
            fp, fn = 0, 0
            if pred[0]:
                tp = confusion[0][0]
                tn = 0
            else:
                tn = confusion[0][0]
                tp = 0
        else:
            tn, fp, fn, tp = confusion.ravel()

        f1_measure = f1_score(outcome, pred, zero_division=0.0)
        logging.info(f"Course ID {crs_id} has {len(crs_df)} students and f1_measure: {f1_measure}")
        results['course_id'].append(crs_id)
        results['f1_measure'].append(f1_measure)
        results['student_cnt'].append(len(crs_df))
        results['accuracy'].append(accuracy)
        results['tn'].append(tn)
        results['fp'].append(fp)
        results['fn'].append(fn)
        results['tp'].append(tp)
    # if return_confusion:
    #     return results, confusion
    return results


def build_and_run_naive_bayes(train_data, test_data):
    train_data = train_data.drop(columns=['user_id', 'course_id', 'user_state'])
    x_train, y_train = train_data.drop('FAIL', axis=1), train_data['FAIL']

    logging.info("\n\nFitting Naive Bayes model")
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    pickle.dump(nb_model, open('best_nb.pkl', 'wb'))

    return gather_f1_measures(nb_model, test_data)


def build_and_run_decision_tree(train_data, test_data):
    train_data = train_data.drop(columns=['user_id', 'course_id', 'user_state'])
    x_train, y_train = train_data.drop('FAIL', axis=1), train_data['FAIL']

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    logging.info("\n\nHypertuning Decision Tree model")
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    best_tree = grid_search.best_estimator_
    pickle.dump(best_tree, open('best_tree.pkl', 'wb'))

    return gather_f1_measures(best_tree, test_data)


def build_and_run_neural_network(train_data, test_data):
    train_data = train_data.sample(n=40000)
    train_data = train_data.drop(columns=['user_id', 'course_id', 'user_state'])
    x_train, y_train = train_data.drop('FAIL', axis=1), train_data['FAIL']

    logging.info("\n\nFitting Neural Network model")
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    # Hyperparameter Tuning
    param_grid = {
        'hidden_layer_sizes': [(100,), (10,10), (15,), (15,15), (225,)],
        'activation': ['tanh', 'relu', 'logistic', 'identity'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'max_iter': [500, 1000, 1500],
        'shuffle': [True, False],
        'tol': [0.0001, 0.001, 0.01]
    }

    mlp = MLPClassifier(random_state=42)
    # model_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
    model_search = RandomizedSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=1, n_iter=25)
    model_search.fit(X_train_scaled, y_train)

    # Train the model with the best hyperparameters
    best_mlp = model_search.best_estimator_
    pickle.dump(best_mlp, open('best_mlp.pkl', 'wb'))

    return gather_f1_measures(best_mlp, test_data, scaler=scaler)


def build_and_run_SVM(train_data, test_data):
    train_data = train_data.sample(n=40000)
    train_data = train_data.drop(columns=['user_id', 'course_id', 'user_state'])
    x_train, y_train = train_data.drop('FAIL', axis=1), train_data['FAIL']

    logging.info("\n\nFitting SVM model")
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    # Hyperparameter Tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],  # only relevant for 'poly' kernel
        'shrinking': [True, False]
    }

    svm = SVC()
    # model_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
    model_search = RandomizedSearchCV(svm, param_grid, cv=2, n_jobs=-1, verbose=1, n_iter=50)
    model_search.fit(X_train_scaled, y_train)

    # Train the model with the best hyperparameters
    best_svm = model_search.best_estimator_
    pickle.dump(best_svm, open('best_svm.pkl', 'wb'))
    # best_svm = SVC(C=100, degree=2, gamma='auto', kernel='poly', shrinking=True, probability=False)
    # best_svm.fit(X_train_scaled, y_train)
    return gather_f1_measures(best_svm, test_data, scaler=scaler)


def build_all_campus_dataframe(data_config_file='data_config.json'):
    header_list = ["user_id", "course_id", "total_activity_time", "user_state", "term_id", "crs_count", "avg_crs_level",
                   "avg_crs_credits", "grade", "adjusted_grade", "submission_day_diff", "pg_level", "pg_normed",
                   "pt_level", "pt_normed", "late_percent", "on_time_percent", "missing_percent", "FAIL", "PASS"]
    with open(data_config_file) as f:
        data_config = json.load(f)

    all_train_data = pd.DataFrame()
    all_test_data = pd.DataFrame()

    for campus in data_config:
        with(open(f'{campus["name"]}.pkl', 'rb')) as f:
            df = pickle.load(f)

        df = [list(item) for item in df]
        df = pd.DataFrame(df, columns=header_list)
        df['PASS'], df['FAIL'] = df['FAIL'], df['PASS']
        train_data, test_data = build_test_and_train_data(df, campus['term_ids'])

        logging.info(f'{campus["name"]} has {len(train_data)} train records and {len(test_data)} test records')
        all_train_data = pd.concat([all_train_data, train_data], ignore_index=True)
        all_test_data = pd.concat([all_test_data, test_data], ignore_index=True)

    all_train_data.drop(columns=['PASS'], inplace=True)
    all_test_data.drop(columns=['PASS'], inplace=True)

    return all_train_data, all_test_data
