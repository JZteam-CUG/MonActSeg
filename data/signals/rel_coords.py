import numpy as np


def get_relative_coordinates(sample, references=(0)):
    # input: C, T, V, M
    org_sample = sample
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    # Shape: C, t, V, M
    ref_loc = sample[:, :, references, :]
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
    rel_coords = np.vstack([coords_diff])

    final_sample[:, start:end, :, :] = rel_coords

    return final_sample
