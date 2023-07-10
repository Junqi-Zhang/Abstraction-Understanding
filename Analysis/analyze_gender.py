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
        if 'female' in n:
            gender = 'female'
            n = n.replace('female', 'gender')
        elif 'male' in n:
            gender = 'male'
            n = n.replace('male', 'gender')
        else:
            continue

        tags = '_'.join(np.sort(n.split('_')[:-1]))
        if tags not in paired_f_dict.keys():
            paired_f_dict[tags] = {'male': [], 'female': []}
        paired_f_dict[tags][gender].append(paired_features[i])

    pop_list = []
    for key in paired_f_dict.keys():
        if (len(paired_f_dict[key]['male']) == 0) or (len(paired_f_dict[key]['female']) == 0):
            pop_list.append(key)
    if len(pop_list) > 0:
        for item in pop_list:
            paired_f_dict.pop(item)

    male_features = np.zeros((len(paired_f_dict.keys()), ndim))
    female_features = np.zeros((len(paired_f_dict.keys()), ndim))
    for i, key in enumerate(paired_f_dict.keys()):
        male_features[i] = np.mean(paired_f_dict[key]['male'], axis=0)
        female_features[i] = np.mean(paired_f_dict[key]['female'], axis=0)

    # find the gender direction
    mean_features = (male_features + female_features) / 2
    male_centered_features = male_features - mean_features
    female_centered_features = female_features - mean_features

    pca = PCA(n_components=5)
    y = pca.fit_transform(np.vstack([male_centered_features, female_centered_features]))
    print(pca.explained_variance_ratio_)
    gender_direction = pca.components_[0]
    print('male')
    print(y[:male_centered_features.shape[0], 0])
    print('female')
    print(y[male_centered_features.shape[0]:, 0])
    print('length')
    print(male_centered_features.shape[0])

    # get mean class center
    categorized_unique_labels = np.unique(categorized_labels)
    categorized_feature_centers = np.zeros((len(categorized_unique_labels), ndim))
    for i, n in enumerate(categorized_unique_labels):
        categorized_feature_centers[i] = np.mean(categorized_features[categorized_labels == n], axis=0)

    gender_scores = np.dot(categorized_feature_centers, gender_direction)

    # save the results
    labels = list(categorized_unique_labels) + ['male_avg', 'female_avg']
    gender_scores = list(gender_scores) + [np.dot(male_features, gender_direction).mean(),
                                           np.dot(female_features, gender_direction).mean()]

    result = pd.DataFrame({'label': labels, 'gender_score_%s' % use_model: gender_scores})
    return result


use_model = 'vit'
use_paired_data = 'paired'
vit_paired = load_and_anlyze(use_model, use_paired_data)

use_model = 'clip'
use_paired_data = 'paired'
clip_paired = load_and_anlyze(use_model, use_paired_data)

paired_result = vit_paired.merge(clip_paired, on='label')
paired_result.to_excel('./output/%s_class_gender_score.xlsx' % use_paired_data, index=False)

use_model = 'vit'
use_paired_data = 'phase'
vit_phase = load_and_anlyze(use_model, use_paired_data)

use_model = 'clip'
use_paired_data = 'phase'
clip_phase = load_and_anlyze(use_model, use_paired_data)

phase_result = vit_phase.merge(clip_phase, on='label')
phase_result.to_excel('./output/%s_class_gender_score.xlsx' % use_paired_data, index=False)
