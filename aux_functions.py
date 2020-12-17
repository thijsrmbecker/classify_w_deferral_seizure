import numpy as np 
import time
import sortednp as snp
from numba import jit
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def load_with(fl_name):
    with open(fl_name, 'rb') as fl:
        arr = np.load(fl, allow_pickle=True)
    return arr


def save_with(fl_name, arr_tosave):
    with open(fl_name, 'wb') as fl:
        arr = np.save(fl, arr_tosave)


def save_with_pickle(fl_name, dict_tosave):
    with open(fl_name, 'wb') as handle:
        pickle.dump(dict_tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)


def open_with_pickle(fl_name):
    with open(fl_name, 'rb') as handle:
        dict_load = pickle.load(handle)
    return dict_load


def save_performance_arrays(str_extension, PPV_store, sens_store, F1_store, detdelay_store, FPrate_store, FP_times_store, TP_store, FN_store, FP_store, dict_seizures_detected):
    results_array_dir = '/final_result_arrays/'
    save_with_pickle(results_array_dir + 'PPV_' + str_extension + '.pkl', PPV_store)
    save_with_pickle(results_array_dir + 'sensitivity_'  + str_extension + '.pkl', sens_store)
    save_with_pickle(results_array_dir + 'F1_' + str_extension + '.pkl', F1_store)
    save_with_pickle(results_array_dir + 'detdelay_' + str_extension + '.pkl', detdelay_store)
    save_with_pickle(results_array_dir + 'FPrate_' + str_extension + '.pkl', FPrate_store)
    save_with_pickle(results_array_dir + 'FPtimes_' + str_extension + '.pkl', FP_times_store)
    save_with_pickle(results_array_dir + 'TP_' + str_extension + '.pkl', TP_store)
    save_with_pickle(results_array_dir + 'FN_' + str_extension + '.pkl', FN_store)
    save_with_pickle(results_array_dir + 'FP_' + str_extension + '.pkl', FP_store)
    save_with_pickle(results_array_dir + 'detected_seizures_' + str_extension + '.pkl', dict_seizures_detected)


@jit(nopython=True)
def calc_idx_to_check_for_seizure(idx_arr, min_n_ones):
    """
    calculate all start indices of EEG segments that could
    be a seizure, even if filtering trust scores.
    The minimum number of ones depends on the detection rule
    """
    length = idx_arr.shape[0]
    start_idx_arr = []
    for i in range(length-10):
        sa = np.sum(idx_arr[i:i+10])
        if sa >= min_n_ones:
            start_idx_arr.append(i)
    return start_idx_arr

    
def calculate_seizure_flags(label_input):
    """ calculate end time of the seizures detected by the classifier """
    label_input = label_input.astype(int)
    label_size = label_input.shape[0]
    seizure_flags = []
    idx_start_segment = 0
    while label_input[idx_start_segment:].shape[0] > 10:
        labels_segment = label_input[idx_start_segment:idx_start_segment+10]
        if sum(labels_segment) > 7:
            seizure_flags.append(idx_start_segment+9)
        idx_start_segment += 1
    seizure_flags = np.array(seizure_flags)
    nflags = seizure_flags.shape[0]
    flags_to_remove = []
    for i in range(nflags-1):
        if (seizure_flags[i+1] - seizure_flags[i]) == 1:
            flags_to_remove.append(i+1)
    seizure_flags = np.delete(seizure_flags, flags_to_remove)
    return np.array(seizure_flags)


def calculate_seizure_flags_w_trustscores_fast(label_input, trust_scores, percentile_level, idx_to_check):
    """ 
    calculate end time of the seizures detected by the classifier by only taking
    the highest percentile_level trust scores into account
    percentile_level is in [0, 100], gives the lowest n^th percentile
    """
    # trust score preprocessing
    high_confidence_points = np.where(trust_scores >= np.percentile(trust_scores, percentile_level))[0] 
    high_confidence_points = high_confidence_points.astype(int)
    label_input = label_input.astype(int)
    # idx_to_check = idx_to_check.astype(int)
    seizure_flags = []
    n_samples = label_input.shape[0]
    n_less_half_trusted = 0
    for idx_start_segment in idx_to_check:
        idx_to_classify = np.array(np.arange(idx_start_segment, idx_start_segment+10))
        labels_to_use = snp.intersect(idx_to_classify, high_confidence_points)
        if labels_to_use.shape[0] > 5:
            if np.mean(label_input[labels_to_use]) > 0.7:
                seizure_flags.append(idx_start_segment+9)
        else:
            # take highest 5 rated samples
            # if 4 or 5 indicate a seizure, it is classified as a seizure
            n_less_half_trusted += 1
            idx_to_classify = idx_to_classify.astype(int)
            trust_scores_segment = trust_scores[idx_to_classify]
            labels_segment = label_input[idx_to_classify]
            ind = np.argpartition(trust_scores_segment, -5)[-5:]
            if sum(labels_segment[ind]) > 3:
                seizure_flags.append(idx_start_segment+9)
    seizure_flags = np.array(seizure_flags)
    nflags = seizure_flags.shape[0]
    flags_to_remove = []
    for i in range(nflags-1):
        if (seizure_flags[i+1] - seizure_flags[i]) == 1:
            flags_to_remove.append(i+1)
    seizure_flags = np.delete(seizure_flags, flags_to_remove)
    return np.array(seizure_flags), n_less_half_trusted


def calculate_seizure_flags_w_trustscores_threshold_fast(label_input, trust_scores, threshold_value, idx_to_check):
    """ 
    calculate end time of the seizures detected by the classifier by only taking
    the highest percentile_level trust scores into account
    percentile_level is in [0, 100], gives the lowest n^th percentile
    """
    # trust score preprocessing
    high_confidence_points = np.where(trust_scores >= threshold_value)[0] 
    high_confidence_points = high_confidence_points.astype(int)
    label_input = label_input.astype(int)
    seizure_flags = []
    n_samples = label_input.shape[0]
    n_less_half_trusted = 0
    for idx_start_segment in idx_to_check:
        idx_to_classify = np.array(np.arange(idx_start_segment, idx_start_segment+10))
        labels_to_use = snp.intersect(idx_to_classify, high_confidence_points)     
        if labels_to_use.shape[0] > 5:
            if np.mean(label_input[labels_to_use]) > 0.7:
                seizure_flags.append(idx_start_segment+9)
        else:
            # take highest 5 rated samples
            # if 4 or 5 indicate a seizure, it is classified as a seizure
            n_less_half_trusted += 1
            trust_scores_segment = trust_scores[idx_to_classify]
            labels_segment = label_input[idx_to_classify]
            ind = np.argpartition(trust_scores_segment, -5)[-5:]
            if sum(labels_segment[ind]) > 3:
                seizure_flags.append(idx_start_segment+9)
    seizure_flags = np.array(seizure_flags)
    nflags = seizure_flags.shape[0]
    flags_to_remove = []
    for i in range(nflags-1):
        if (seizure_flags[i+1] - seizure_flags[i]) == 1:
            flags_to_remove.append(i+1)
    seizure_flags = np.delete(seizure_flags, flags_to_remove)
    return np.array(seizure_flags), n_less_half_trusted


def calculate_real_seizure_timings(label_input):
    """
    calculate the start and end times (+1, as in Kaat her code) of the real seizures
    """
    where_one = np.where(label_input == 1)[0]
    seizure_timings = []
    n_elements = where_one.shape[0]
    if n_elements == 0:
        return np.array([])
    else:
        seizure_start = where_one[0]
        seizure_end = 0   
        # initiated to avoid the error:
        # UnboundLocalError: local variable 'seizure_end' referenced before assignment
        for i in range(1, n_elements): 
            if where_one[i] - where_one[i-1] > 1:
                seizure_end = where_one[i-1]+1
                seizure_timings.append([seizure_start, seizure_end])
                seizure_start = where_one[i] 
        if seizure_end != where_one[-1]:
            seizure_timings.append([seizure_start, where_one[-1]+1])
        return np.array(seizure_timings)  


def precision_metrics_kaat_10s(seizure_flags, seizure_timings):
    """
    seizure_flags: end of 10-second seizure detection by the algorithm
    seizure_timings: start and end time of real seizures
    The end of the seizure flag from the classifier should fall in the real seizure interval
    false positives are calculated with a 10 second interval between them
    """
    TP = 0
    FP = 0
    FN = 0
    det_delay = []
    total_seizure_timings = seizure_timings.shape[0]
    seizure_timings_detected = []
    cnt = 0
    FP_times = []
    # calculate false and true positives
    for flag_timing in seizure_flags:
        TP_detected = False
        for i in range(total_seizure_timings):
            if (flag_timing >= seizure_timings[i][0]) and (flag_timing <= seizure_timings[i][1]+1):
                if i not in seizure_timings_detected:
                    TP += 1
                    seizure_timings_detected.append(i)
                    det_delay.append(flag_timing - seizure_timings[i][0])
                TP_detected = True
        if not TP_detected:
            FP_times.append(flag_timing)
    # calculate false positives
    n_fpt = len(FP_times)
    fp_times_final = []
    cnt = 0
    if n_fpt > 0:
        last_fp = FP_times[0]
        FP += 1
        fp_times_final.append(last_fp)
        for cnt in range(1, n_fpt):
            fpt = FP_times[cnt]
            if abs(fpt - last_fp) > 10:
                FP += 1
                last_fp = fpt
                fp_times_final.append(fpt)
    # calculate false negatives
    for seizure in seizure_timings:
        seizure_detected = False
        for flags in seizure_flags:
            if (flags >= seizure[0]) and (flags <= seizure[1]+9):
                seizure_detected = True
        if not seizure_detected:
            FN += 1
    if (total_seizure_timings > 0) and (len(seizure_timings_detected) > 0):
        detected_seizure_timings = seizure_timings[seizure_timings_detected]
    else:
        detected_seizure_timings = []
    return TP, FP, FN, total_seizure_timings, det_delay, detected_seizure_timings, fp_times_final


def print_summary_statistics(perf_arr, Zdsn):
    print("mean + std: ")
    mean = np.nanmean(perf_arr)
    # print(mean)
    print(str.format('{0:.3f}', mean))
    std = np.nanstd(perf_arr)   #  * Zdsn
    # print(std)
    print(str.format('{0:.3f}', std))
    print("median + range:")
    median = np.nanmedian(perf_arr) 
    # print(median)
    print(str.format('{0:.3f}', median))
    range_min = np.nanmin(perf_arr) 
    range_max = np.nanmax(perf_arr)
    # print(str(range_min) + " -- " + str(range_max))
    print(str.format('{0:.3f}', range_min) + " -- " + str.format('{0:.3f}', range_max))
    print("")
    return mean, std, median, range_min, range_max


def sigmoid_trust(x, a=1.):
    platt_val = []
    for el in x:
        platt_val.append(1. / (1. + np.exp(- a * el)))
    trust_platt = []
    for el in platt_val:
        if el >= 0.5:
            trust_platt.append(abs((el - 0.5) * 2.))
        else:
            trust_platt.append(abs((0.5 - el) * 2.))
    return trust_platt
