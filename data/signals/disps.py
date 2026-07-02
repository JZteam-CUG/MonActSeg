import numpy as np


def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end - 1, :, :] = disps

    return final_sample


def get_velocity(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    if t < 2:
        return final_sample

    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    final_sample[:, start:end - 1, :, :] = disps

    return final_sample


def get_acceleration(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    valid_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = valid_frames.argmax()
    end = len(valid_frames) - valid_frames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    if t < 3:
        return final_sample

    velocity = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]

    final_sample[:, start:end - 2, :, :] = acceleration

    return final_sample
