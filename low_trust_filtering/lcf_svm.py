import sys
sys.path.insert(1, '../')
import os.path
import numpy as np 
import time
import aux_functions as af


def concat_trust_score(dict_trust, pat_to_remove, n_pat, pid_arr):
    """
    concatenate all trust scores, expect for pat_to_remove
    pat_to_remove is the index in the patient_id array, not the patient_id itself
    """
    all_trust_scores = []
    for pat_cnt in range(n_pat):
        if pat_cnt != pat_to_remove:
            pid = pid_arr[pat_cnt]
            pid_str = str(pid)
            all_trust_scores = all_trust_scores + list(dict_trust[pid_str])
    return all_trust_scores


def return_svm_trust_scores(a_scaling, pat_id_arr, aimII, tscaling):
    dict_trust = {}
    if aimII:
        scores_svm_dir = '/SVM_confidence_aimII/'
        svm_string = 'svm_aimII_scores_pat_'
    else:
        scores_svm_dir = '/SVM_confidence/'
        svm_string = 'svm_scores_pat_'
    for pid in pat_id_arr:
        pid_str = str(pid)
        ts_filename = scores_svm_dir + svm_string + str(pid) + '.npy'
        trust_scores_pat = af.load_with(ts_filename)
        # do temperature scaling with parameter a
        if tscaling:
            dict_trust[pid_str] = af.sigmoid_trust(trust_scores_pat, a=a_scaling)
        else:
            dict_trust[pid_str] = abs(trust_scores_pat)
    return dict_trust


# directories containing the data
labels_dir_base = '/Annotations_outputs_classifier/'

# patient IDs with measurements
pat_id_arr_original = [10,12,13,14,15,16,17,22,24,30,33,34,36,40,41,54,56,59,61,63,64,65,67,68,70,71,72,73,74,75,
                       76,77,78,79,82,83,89,90,93,94,95,99]
pat_id_arr_new = [11, 25, 28, 35, 47, 48, 58, 60, 66, 80, 92, 98]
pat_id_arr = pat_id_arr_original + pat_id_arr_new
n_pat = len(pat_id_arr)
pat_id_arr = np.array(pat_id_arr)

F1_final = []
PPV_final = []
detection_sensitivity_final = []
FP_rate_final = []
not_trusted_final = []
detection_delay_final = []

det_sens_perc_av = []
det_sens_perc_std = []
det_sens_perc_med = []
det_sens_perc_range = []

FP_perc_av = []
FP_perc_std = []
FP_perc_med = []
FP_perc_range = []

F1_perc_av = []
F1_perc_std = []
F1_perc_med = []
F1_perc_range = []

PPV_perc_av = []
PPV_perc_std = []
PPV_perc_med = []
PPV_perc_range = [] 

dd_perc_av = []
dd_perc_std = []
dd_perc_med = []
dd_perc_range = []

# use the SVM model trained on the aimII labels or not (= SVM trained on aimII labels)
AIMII_SVM_MODEL = False
# use threshold strategy (patient-specific percentile) or not (= patient-independent percentile)
THRESHOLD = False
# use temperature scaling or consider the distances to the decision boundary
TEMPERATURE_SCALING = False
per_arr_cv = [0.01, 0.1, 0.5, 1., 2., 5., 10., 20., 40., 80.]
if TEMPERATURE_SCALING:
    a_vals = [0.01, 0.05, 0.1, 0.2, 0.5, 1., 2.]
else:
    a_vals = [0]

n_percent_cv = len(per_arr_cv)
n_a_vals = len(a_vals)

time_start_all = time.time()

# load all numpy files at once
dict_labels = {}
dict_seizure_timings = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    # note that this should actually have been called aim2
    fln_labels_all = labels_dir_base + 'Patient_labelsaim1_' + str(pid) + '.npy'
    labels_all = af.load_with(fln_labels_all)
    dict_labels[pid_str] = labels_all
    dict_seizure_timings[pid_str] = af.calculate_real_seizure_timings(labels_all)
dict_predictions = {}
if AIMII_SVM_MODEL:
    str_predictions = 'Patient_predictions_aimII_rmsa_'
else:
    str_predictions = 'Patient_predictions_rmsa_'
for pid in pat_id_arr:
    pid_str = str(pid)
    fln_predictions_all = labels_dir_base + str_predictions + str(pid) + '.npy'
    dict_predictions[pid_str] = af.load_with(fln_predictions_all)
dict_idx_to_check = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    predictions_all = dict_predictions[pid_str]
    idx_to_check = af.calc_idx_to_check_for_seizure(predictions_all, min_n_ones=4)
    dict_idx_to_check[pid_str] = idx_to_check

# calculate the performance for each patient / percentage combination
F1_pat_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
F1_pat_per[:, :, :] = np.nan
PPV_pat_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
PPV_pat_per[:, :, :] = np.nan
sens_pat_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
sens_pat_per[:, :, :] = np.nan
FP_pat_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
FP_pat_per[:, :, :] = np.nan
det_delay_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
det_delay_per[:, :, :] = np.nan
FP_cnt_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
FP_cnt_per[:, :, :] = np.nan
FN_cnt_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
FN_cnt_per[:, :, :] = np.nan
TP_cnt_per = np.zeros((n_pat, n_percent_cv, n_a_vals))
TP_cnt_per[:, :, :] = np.nan

sens_store = {}
F1_store = {}
PPV_store = {}
detdelay_store = {}
FPrate_store = {}
TP_store = {}
FP_store = {}
FP_times_store = {}
FN_store = {}
dict_seizures_detected = {}

for a_cnt in range(n_a_vals):
    aval = a_vals[a_cnt]
    dict_trust = return_svm_trust_scores(aval, pat_id_arr, aimII=AIMII_SVM_MODEL, tscaling=TEMPERATURE_SCALING)
    for pat_cnt in range(n_pat):
        pid = pat_id_arr[pat_cnt]
        pid_str = str(pid)
        seizure_timings = dict_seizure_timings[pid_str]
        trust_scores_pat = np.array(dict_trust[pid_str])
        predictions_all = dict_predictions[pid_str]
        idx_pat_to_check = dict_idx_to_check[pid_str]
        if THRESHOLD:
            all_trust_scores = concat_trust_score(dict_trust=dict_trust, pat_to_remove=pat_cnt, n_pat=n_pat, pid_arr=pat_id_arr)
        for per_cnt in range(n_percent_cv):
            detection_delay_pat = []
            per = per_arr_cv[per_cnt]
            # calculate seizure flags
            if THRESHOLD:
                threshold_trust_scores = np.percentile(all_trust_scores, per)
                seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_threshold_fast(label_input=predictions_all,
                                                            trust_scores=trust_scores_pat, threshold_value=threshold_trust_scores, idx_to_check=idx_pat_to_check)
            else:
                seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(label_input=predictions_all, 
                                                                trust_scores=trust_scores_pat, percentile_level=per, idx_to_check=idx_pat_to_check)
            # calculate precision metrics
            TP, FP, FN, total_seizures, det_delay, detected_seizure_timings, FP_times = af.precision_metrics_kaat_10s(seizure_flags, seizure_timings)
            detection_delay_pat = detection_delay_pat + det_delay
            if total_seizures > 0: 
                if int(TP) > 0:
                    detection_sensitivity = TP / (TP + FN)
                    PPV = TP / (TP + FP)
                    F1 = 2.*TP / (2.*TP + FP + FN)
                else:
                    detection_sensitivity = 0.
                    PPV = 0.
                    F1 = 0.
                F1_pat_per[pat_cnt, per_cnt, a_cnt] = F1
                PPV_pat_per[pat_cnt, per_cnt, a_cnt] = PPV
                sens_pat_per[pat_cnt, per_cnt, a_cnt] = detection_sensitivity
                det_delay_per[pat_cnt, per_cnt, a_cnt] = np.nanmean(detection_delay_pat)
                TP_cnt_per[pat_cnt, per_cnt, a_cnt] = TP
                FN_cnt_per[pat_cnt, per_cnt, a_cnt] = FN
            n_hours = float(labels_all.shape[0]) / (60. * 60.)
            FP_rate = 24. * FP / n_hours
            FP_pat_per[pat_cnt, per_cnt, a_cnt] = FP_rate
            FP_cnt_per[pat_cnt, per_cnt, a_cnt] = FP
        
pat_cnt_arr = np.arange(n_pat)
max_per_arr = []
max_a_arr = []
for pat_cnt in range(n_pat):
    # remove patient from F1 array
    pat_to_keep = np.delete(pat_cnt_arr, pat_cnt)
    pid = pat_id_arr[pat_cnt]
    pid_str = str(pid)
    F1_cv = F1_pat_per[pat_to_keep, :, :]
    # take percentage with maximum average
    F1_av = np.nanmean(F1_cv, axis=0)
    max_ind = np.unravel_index(np.argmax(F1_av, axis=None), F1_av.shape)
    max_per = per_arr_cv[max_ind[0]]
    max_per_arr.append(max_per)
    max_a = a_vals[max_ind[1]]
    max_a_arr.append(max_a)
    F1_final.append(F1_pat_per[pat_cnt, max_ind[0], max_ind[1]])
    F1_store[pid_str] = F1_pat_per[pat_cnt, max_ind[0], max_ind[1]]
    PPV_final.append(PPV_pat_per[pat_cnt, max_ind[0], max_ind[1]])
    PPV_store[pid_str] = PPV_pat_per[pat_cnt, max_ind[0], max_ind[1]]
    detection_sensitivity_final.append(sens_pat_per[pat_cnt, max_ind[0], max_ind[1]])
    sens_store[pid_str] = sens_pat_per[pat_cnt, max_ind[0], max_ind[1]]
    FP_rate_final.append(FP_pat_per[pat_cnt, max_ind[0], max_ind[1]])
    FPrate_store[pid_str] = FP_pat_per[pat_cnt, max_ind[0], max_ind[1]]
    detection_delay_final.append(det_delay_per[pat_cnt, max_ind[0], max_ind[1]])
    detdelay_store[pid_str] = det_delay_per[pat_cnt, max_ind[0], max_ind[1]]
    TP_store[pid_str] = TP_cnt_per[pat_cnt, max_ind[0], max_ind[1]]
    FP_store[pid_str] = FP_cnt_per[pat_cnt, max_ind[0], max_ind[1]]
    FN_store[pid_str] = FN_cnt_per[pat_cnt, max_ind[0], max_ind[1]]
    
# make data for plots
Z_val = 1.96
n_estimates = n_pat
sqrt_n = np.sqrt(n_estimates)
Zdsn = Z_val / sqrt_n

print("detection sensitivity: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(detection_sensitivity_final, Zdsn)
det_sens_perc_av.append(mn)
det_sens_perc_std.append(st)
det_sens_perc_med.append(med)
det_sens_perc_range.append([rmin, rmax])

print("FP rate: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(FP_rate_final, Zdsn)
FP_perc_av.append(mn)
FP_perc_std.append(st)
FP_perc_med.append(med)
FP_perc_range.append([rmin, rmax])

print("PPV: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(PPV_final, Zdsn)
PPV_perc_av.append(mn)
PPV_perc_std.append(st)
PPV_perc_med.append(med)
PPV_perc_range.append([rmin, rmax])

print("F1: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(F1_final, Zdsn)
F1_perc_av.append(mn)
F1_perc_std.append(st)
F1_perc_med.append(med)
F1_perc_range.append([rmin, rmax])

print('detection delay: ')
mn, st, med, rmin, rmax = af.print_summary_statistics(detection_delay_final, Zdsn)
dd_perc_av.append(mn)
dd_perc_std.append(st)
dd_perc_med.append(med)
dd_perc_range.append([rmin, rmax])


print("# detection sensitivity: ")
print('sens = ' + str(det_sens_perc_av))
print('sens_std = ' + str(det_sens_perc_std))
print('sens_med = ' + str(det_sens_perc_med))
print('sens_range = ' + str(det_sens_perc_range))

print("# FP rate: ")
print('FP = ' + str(FP_perc_av))
print('FP_std = ' + str(FP_perc_std))
print('FP_med = ' + str(FP_perc_med))
print('FP_range = ' + str(FP_perc_range))

print("# PPV: ")
print('PPV = ' + str(PPV_perc_av))
print('PPV_std = ' + str(PPV_perc_std))
print('PPV_med = ' + str(PPV_perc_med))
print('PPV_range = ' + str(PPV_perc_range))

print("# F1: ")
print('F1 = ' + str(F1_perc_av))
print('F1_std = ' + str(F1_perc_std))
print('F1_med = ' + str(F1_perc_med))
print('F1_range = ' + str(F1_perc_range))

print("# detection delay: ")
print('dd = ' + str(dd_perc_av))
print('dd_std = ' + str(dd_perc_std))
print('dd_med = ' + str(dd_perc_med))
print('dd_range = ' + str(dd_perc_range))

print("percent removed: ")
print(np.mean(max_per_arr))

print("a used: ")
print(np.mean(max_a_arr))

time_end_all = time.time()
print("total calculation time: " + "%.2f" % ((time_end_all - time_start_all) / 60.) + ' minutes')
print('')

if THRESHOLD:
    str_extension_1 = 'svm_threshold'
else:
    str_extension_1 = 'svm_percentile'
if TEMPERATURE_SCALING:
    str_extension_2 = '_temp'
else:
    str_extension_2 = '_no_temp'
if AIMII_SVM_MODEL:
    str_extension_3 = '_AIMII'
    str_extension = str_extension_1 + str_extension_2 + str_extension_3
else:
    str_extension = str_extension_1 + str_extension_2

results_array_dir = '/final_result_arrays/'
# af.save_with_pickle(results_array_dir + 'PPV_' + str_extension + '.pkl', PPV_store)
# af.save_with_pickle(results_array_dir + 'sensitivity_'  + str_extension + '.pkl', sens_store)
# af.save_with_pickle(results_array_dir + 'F1_' + str_extension + '.pkl', F1_store)
# af.save_with_pickle(results_array_dir + 'detdelay_' + str_extension + '.pkl', detdelay_store)
# af.save_with_pickle(results_array_dir + 'FPrate_' + str_extension + '.pkl', FPrate_store)
# af.save_with_pickle(results_array_dir + 'TP_' + str_extension + '.pkl', TP_store)
# af.save_with_pickle(results_array_dir + 'FN_' + str_extension + '.pkl', FN_store)
# af.save_with_pickle(results_array_dir + 'FP_' + str_extension + '.pkl', FP_store)
