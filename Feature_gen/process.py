import numpy as np
from imagenet_classes import IMAGENET2012_CLASSES

# process Caltech-256
data = np.load('./temp/caltech-256_vit-base-patch16-224-in21k.npz')
vit_file_names = data['file_names']
vit_features = data['features']


data = np.load('./temp/caltech-256_clip-vit-base-patch16.npz')
clip_file_names = data['file_names']
clip_features = data['features']

for n1, n2 in zip(vit_file_names, clip_file_names):
    assert n1 == n2

file_names = clip_file_names
labels = np.array([n.split('/')[-2] for n in file_names])

np.savez('caltech-256_features.npz', file_names=file_names,
         vit_features=vit_features, clip_features=clip_features, labels=labels)

# process paired
data = np.load('./temp/paired_vit-base-patch16-224-in21k.npz')
vit_file_names = data['file_names']
vit_features = data['features']


data = np.load('./temp/paired_clip-vit-base-patch16.npz')
clip_file_names = data['file_names']
clip_features = data['features']

for n1, n2 in zip(vit_file_names, clip_file_names):
    assert n1 == n2

file_names = clip_file_names
labels = np.array(['_'.join(n.split('/')[-1].split('_')[:-1]) for n in file_names])

np.savez('paired_features.npz', file_names=file_names,
         vit_features=vit_features, clip_features=clip_features, labels=labels)

# process categoried
data = np.load('./temp/categorized_vit-base-patch16-224-in21k.npz')
vit_file_names = data['file_names']
vit_features = data['features']


data = np.load('./temp/categorized_clip-vit-base-patch16.npz')
clip_file_names = data['file_names']
clip_features = data['features']

for n1, n2 in zip(vit_file_names, clip_file_names):
    assert n1 == n2

file_names = clip_file_names
labels = np.array(['_'.join(n.split('/')[-1].split('_')[:-1]) for n in file_names])

np.savez('categorized_features.npz', file_names=file_names,
         vit_features=vit_features, clip_features=clip_features, labels=labels)

# process ImageNet
data = np.load('./temp/imagenet_vit-base-patch16-224-in21k.npz')
vit_file_names = data['file_names']
vit_features = data['features']


data = np.load('./temp/imagenet_clip-vit-base-patch16.npz')
clip_file_names = data['file_names']
clip_features = data['features']

for n1, n2 in zip(vit_file_names, clip_file_names):
    assert n1 == n2

file_names = clip_file_names
labels = np.array([n.split('_')[-1].split('.')[0] for n in file_names])

vit_feat_center = []
clip_feat_center = []
label_center = []
for key, value in IMAGENET2012_CLASSES.items():
    vit_feat_center.append(np.mean(vit_features[labels == key], axis=0))
    clip_feat_center.append(np.mean(clip_features[labels == key], axis=0))
    label_center.append(value)

vit_feat_center = np.vstack(vit_feat_center)
clip_feat_center = np.vstack(clip_feat_center)
label_center = np.array(label_center)

np.savez('imagenet_features.npz', file_names=np.array([]),
         vit_features=vit_feat_center, clip_features=clip_feat_center, labels=label_center)

# process phase
data = np.load('./temp/phase_vit-base-patch16-224-in21k.npz')
vit_file_names = data['file_names']
vit_features = data['features']


data = np.load('./temp/phase_clip-vit-base-patch16.npz')
clip_file_names = data['file_names']
clip_features = data['features']

for n1, n2 in zip(vit_file_names, clip_file_names):
    assert n1 == n2

file_names = clip_file_names
labels = np.array(['_'.join(n.split('/')[-1].split('_')[:-1]) for n in file_names])

np.savez('phase_features.npz', file_names=file_names,
         vit_features=vit_features, clip_features=clip_features, labels=labels)
