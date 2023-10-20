import numpy as np
import pandas as pd



if __name__ == '__main__':
    nb_df = pd.read_csv('nb_results.csv')
    tree_df = pd.read_csv('tree_results.csv')
    nn_df = pd.read_csv('nn_results.csv')
    svm_df = pd.read_csv('svm_results.csv')

    nb_f1 = nb_df['f1_measure'].tolist()
    tree_f1 = tree_df['f1_measure'].tolist()
    nn_f1 = nn_df['f1_measure'].tolist()
    svm_f1 = svm_df['f1_measure'].tolist()

    group_names = ['Naive Bayes', 'Decision Tree', 'Neural Network', 'SVM']
    groups = [nb_f1, tree_f1, nn_f1, svm_f1]

    print(f'Model\tMean\tStd Dev\tMin\tMax')
    for name, group in zip(group_names, groups):
        print(f'{name}\t{np.mean(group)}\t{np.std(group)}\t{np.min(group)}\t{np.max(group)}')
