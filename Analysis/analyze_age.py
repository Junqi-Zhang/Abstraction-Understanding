import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ndim = 768


def load_and_anlyze(use_model, use_paired_data):
    # load data
    paired_data = np.load('./feature/%s_features.npz' % use_paired_data)
    caltech_data = np.load('./feature/caltech-256_features.npz')
    categorized_data = np.load('./feature/categorized_features.npz')

    paired_labels = paired_data['labels']
    categorized_labels = np.concatenate([caltech_data['labels'], categorized_data['labels']])

    if use_model == 'clip':
        paired_features = paired_data['clip_features']
        categorized_features = np.vstack(
            [caltech_data['clip_features'], categorized_data['clip_features']])
    elif use_model == 'vit':
        paired_features = paired_data['vit_features']
        categorized_features = np.vstack(
            [caltech_data['vit_features'], categorized_data['vit_features']])
    else:
        raise NotImplementedError

    # process paired features
    paired_f_dict = dict()
    for i, n in enumerate(paired_labels):
        if n.startswith('baby'):
            age = 'junior'
            n = n.replace('baby', 'age')
        elif n.startswith('children'):
            age = 'junior'
            n = n.replace('children', 'age')
        elif n.startswith('young'):
            age = 'middle'
            n = n.replace('young', 'age')
        elif n.startswith('adult'):
            age = 'middle'
            n = n.replace('adult', 'age')
        elif n.startswith('senior'):
            age = 'senior'
            n = n.replace('senior', 'age')
        else:
            continue

        tags = '_'.join(np.sort(n.split('_')[:-1]))
        if tags not in paired_f_dict.keys():
            paired_f_dict[tags] = {'junior': [], 'middle': [], 'senior': []}
        paired_f_dict[tags][age].append(paired_features[i])

    junior_features = []
    senior_features = []
    for key in paired_f_dict.keys():
        if len(paired_f_dict[key]['junior']) > 0 and len(paired_f_dict[key]['middle']) > 0:
            junior_features.append(np.mean(paired_f_dict[key]['junior'], axis=0))
            senior_features.append(np.mean(paired_f_dict[key]['middle'], axis=0))
        if len(paired_f_dict[key]['junior']) > 0 and len(paired_f_dict[key]['senior']) > 0:
            junior_features.append(np.mean(paired_f_dict[key]['junior'], axis=0))
            senior_features.append(np.mean(paired_f_dict[key]['senior'], axis=0))
        if len(paired_f_dict[key]['middle']) > 0 and len(paired_f_dict[key]['senior']) > 0:
            junior_features.append(np.mean(paired_f_dict[key]['middle'], axis=0))
            senior_features.append(np.mean(paired_f_dict[key]['senior'], axis=0))
    junior_features = np.vstack(junior_features)
    senior_features = np.vstack(senior_features)

    # find the gender direction
    mean_features = (junior_features + senior_features) / 2
    senior_centered_features = senior_features - mean_features
    junior_centered_features = junior_features - mean_features

    pca = PCA(n_components=5)
    y = pca.fit_transform(np.vstack([senior_centered_features, junior_centered_features]))
    print(pca.explained_variance_ratio_)
    age_direction = pca.components_[0]
    print('male')
    print(y[:senior_centered_features.shape[0], 0])
    print('female')
    print(y[senior_centered_features.shape[0]:, 0])
    print('length')
    print(senior_centered_features.shape[0])

    # get mean class center
    categorized_unique_labels = np.unique(categorized_labels)
    categorized_feature_centers = np.zeros((len(categorized_unique_labels), ndim))
    for i, n in enumerate(categorized_unique_labels):
        categorized_feature_centers[i] = np.mean(categorized_features[categorized_labels == n], axis=0)

    age_scores = np.dot(categorized_feature_centers, age_direction)

    # save the results
    labels = list(categorized_unique_labels) + ['senior_avg', 'junior_avg']
    age_scores = list(age_scores) + [np.dot(senior_features, age_direction).mean(),
                                     np.dot(junior_features, age_direction).mean()]

    result = pd.DataFrame({'label': labels, 'age_score_%s' % use_model: age_scores})
    return result


use_model = 'vit'
use_paired_data = 'phase'
vit_phase = load_and_anlyze(use_model, use_paired_data)

use_model = 'clip'
use_paired_data = 'phase'
clip_phase = load_and_anlyze(use_model, use_paired_data)

phase_result = vit_phase.merge(clip_phase, on='label')
phase_result.to_excel('./output/%s_class_age_score.xlsx' % use_paired_data, index=False)
