import torch
import numpy as np
import random
from data.signals.disps import get_displacements, get_acceleration
from data.signals.rel_coords import get_relative_coordinates


def get_features(sample):
    disps = get_displacements(sample)
    acceleration = get_acceleration(sample)
    rel_coords = get_relative_coordinates(sample, references=(4))
    sample = np.concatenate([rel_coords, disps, acceleration], axis=0)
    return sample


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, skeleton_features_path, rgb_features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.skeleton_features_path = skeleton_features_path
        self.rgb_features_path = rgb_features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list_file):
        with open(vid_list_file, 'r') as file_ptr:
            self.list_of_examples = file_ptr.read().split('\n')[:-1]
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input_skeleton = []
        batch_input_rgb = []
        batch_target = []
        for vid in batch:
            skeleton_features = np.load(self.skeleton_features_path + vid.split('.')[0] + '.npy')
            skeleton_features = get_features(skeleton_features)
            rgb_features = np.load(self.rgb_features_path + vid.split('.')[0] + '.npy')

            with open(self.gt_path + vid, 'r') as file_ptr:
                content = file_ptr.read().splitlines()
            seq_len = min(np.shape(skeleton_features)[1], np.shape(rgb_features)[1], len(content))
            if seq_len <= 0:
                raise ValueError(f"Empty sequence after alignment: {vid}")
            skeleton_features = skeleton_features[:, :seq_len, :, :]
            rgb_features = rgb_features[:, :seq_len]
            classes = np.zeros(seq_len)
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            batch_input_skeleton.append(skeleton_features[:, ::self.sample_rate])
            batch_input_rgb.append(rgb_features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_skeleton_tensor = torch.zeros(len(batch_input_skeleton), 6, max(length_of_sequences), 17, 1, dtype=torch.float)
        batch_input_rgb_tensor = torch.zeros(len(batch_input_rgb), np.shape(batch_input_rgb[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input_skeleton), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input_skeleton), self.num_classes, max(length_of_sequences), dtype=torch.float)
        sample_weight = torch.ones(len(batch_input_skeleton), max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input_skeleton)):
            batch_input_skeleton_tensor[i, :, :np.shape(batch_input_skeleton[i])[1], :, :] = torch.from_numpy(batch_input_skeleton[i])
            batch_input_rgb_tensor[i, :, :np.shape(batch_input_rgb[i])[1]] = torch.from_numpy(batch_input_rgb[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_skeleton_tensor, batch_input_rgb_tensor, batch_target_tensor, mask, sample_weight
