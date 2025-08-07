import numpy as np
# import scipy.io as sio


def load_srt_de(data, channel_norm,label_type, num_windows):
    # isFilt: False  filten:1   channel_norm: True
    n_subs = 123
    if label_type == 'cls2':
        n_vids = 24
    elif label_type == 'cls9':
        n_vids = 28
    n_samples = np.ones(n_vids).astype(np.int32) * num_windows  # (30,30,...,30)

    n_samples_cum = np.concatenate(
        (np.array([0]), np.cumsum(n_samples)))  # (0,30,60,...,810,840)

    # if label_type == 'cls2':
    #     data = data.reshape([data.shape[0], -1, num_windows, data.shape[2]])
    #     vid_sel = list(range(12))
    #     vid_sel.extend(list(range(16,28)))
    #     data = data[:, vid_sel, :, :] # sub, vid, n_channs, n_points
    #     data = data.reshape([data.shape[0], -1, data.shape[2]])
    #     n_videos = 24
    # else:
    #     n_videos = 28


    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :],
                             axis=0)) / np.std(data[i, :, :], axis=0)

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1, 4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5, 9):
            label.extend([i] * 3)
        # print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return label_repeat
