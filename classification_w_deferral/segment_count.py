import numpy as np 


def count_segments(segments_to_check, total_length):
    n_seg = 0
    seg_length = []
    n_segments_algo = np.array(segments_to_check).shape[0]
    if segments_to_check[0][0] != 0:
        n_seg += 1.
        seg_length.append(segments_to_check[0][0] - 1)
    for i in range(n_segments_algo-1):
        n_seg += 1.
        seg_length.append(segments_to_check[i+1][0] - segments_to_check[i][1] - 1)
    if segments_to_check[-1][1] != (total_length - 1):
        n_seg += 1.
        seg_length.append(total_length - 1 - segments_to_check[-1][1])
    seg_length = np.array(seg_length)
    seg_length = seg_length / 60.
    mean_length = np.mean(seg_length)
    std_length = np.std(seg_length)
    # normalize total number of segments per day
    n_days = total_length / (60. * 60. * 24.)
    n_seg /= n_days
    return n_seg, mean_length, std_length
