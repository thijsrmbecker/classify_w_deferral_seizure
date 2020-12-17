import numpy as np 
import time
import aux_functions as af
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

"""
the final version of the code for the SVM model of the Epilepsia article
"""

start_time = time.time()

# use model trained on the AimII labels (if False: model trained on the AimII train labels)
AIMII_SVM_MODEL = True

# directories containing the labels and the features
labels_dir_base = '/Annotations_outputs_classifier/'
# patient IDs with measurements
pat_id_arr_original = [10,12,13,14,15,16,17,22,24,30,33,34,36,40,41,54,56,59,61,63,64,65,67,68,70,71,72,73,74,75,
                       76,77,78,79,82,83,89,90,93,94,95,99]
pat_id_arr_new = [11, 25, 28, 35, 47, 48, 58, 60, 66, 80, 92, 98]
# number of recordings for each patient
n_rec_arr = [12, 13, 5, 11, 13, 12, 11, 12, 9, 11, 11, 7, 9, 9, 8, 13, 11, 10, 15, 22, 10, 8, 12, 6, 15, 11, 15, 
         13, 11, 11, 6, 6, 7, 9, 16, 14, 5, 8, 11, 14, 14, 12]

pat_id_arr = pat_id_arr_original + pat_id_arr_new
n_pat = len(pat_id_arr)

# dictionaries with predictions per 2-second segment
dict_predictions = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    if AIMII_SVM_MODEL:
        fln_predictions_all = labels_dir_base + 'Patient_predictions_aimII_rmsa_' + str(pid) + '.npy'
    else:
        fln_predictions_all = labels_dir_base + 'Patient_predictions_rmsa_' + str(pid) + '.npy'
    dict_predictions[pid_str] = af.load_with(fln_predictions_all)
# dictionary of start indices of 10-second segments which could possibly be a seizure
# even after filtering trust scores. (This means: at least 4 ones of the 10 labels)
# by only checking these segments, the code is much faster
dict_idx_to_check = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    predictions_all = dict_predictions[pid_str]
    idx_to_check = af.calc_idx_to_check_for_seizure(predictions_all, min_n_ones=4)
    dict_idx_to_check[pid_str] = idx_to_check

# define array to store results
F1_all = []
PPV_all = []
detection_sensitivity_all = []
FP_rate_all = []
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
dict_seizures_all = {}


for cnt in range(n_pat):

    pat_id = pat_id_arr[cnt]
    # note that this should actually have been called aim2
    labels_all = af.load_with(labels_dir_base + 'Patient_labelsaim1_' + str(pat_id) + '.npy')
    pid_str = str(pat_id)
    predictions_all = dict_predictions[pid_str]
    idx_pat_to_check = dict_idx_to_check[pid_str]
    detection_delay_pat = []

    TP = 0
    FP = 0
    FN = 0
    total_seizures = 0

    seizure_timings = af.calculate_real_seizure_timings(labels_all)
    seizure_flags = af.calculate_seizure_flags(predictions_all)
    TP, FP, FN, total_seizures, det_delay, detected_seizure_timings, FP_times = af.precision_metrics_kaat_10s(seizure_flags, seizure_timings)
    detection_delay_pat = detection_delay_pat + det_delay

    if total_seizures > 0: 
        if int(TP) > 0:
            detection_sensitivity = TP / (TP + FN)
            PPV = TP / (TP + FP) 
            F1 = 2.*TP / (2.*TP + FP + FN)
        else:
            detection_sensitivity = 0.
            F1 = 0.
            PPV = 0.
        PPV_all.append(PPV)
        F1_all.append(F1)
        detection_sensitivity_all.append(detection_sensitivity)
        det_delay_av = np.mean(detection_delay_pat)
        detection_delay_final.append(det_delay_av)
    n_hours = float(labels_all.shape[0]) / (60. * 60.)
    FP_rate = 24. * FP / n_hours
    FP_rate_all.append(FP_rate)

    # store final values in array
    if total_seizures > 0:
        PPV_store[pid_str] = PPV
        sens_store[pid_str] = detection_sensitivity
        F1_store[pid_str] = F1
        detdelay_store[pid_str] = np.mean(det_delay)
        TP_store[pid_str] = TP
        FN_store[pid_str] = FN
    else:
        PPV_store[pid_str] = np.nan
        sens_store[pid_str] = np.nan
        F1_store[pid_str] = np.nan
        detdelay_store[pid_str] = np.nan
        TP_store[pid_str] = np.nan
        FN_store[pid_str] = np.nan
    FP_store[pid_str] = FP
    FP_times_store[pid_str] = FP_times
    FPrate_store[pid_str] = FP_rate
    dict_seizures_all[pid_str] = seizure_timings
    dict_seizures_detected[pid_str] = detected_seizure_timings
 

end_time = time.time()
print("calculation time: " + "%.2f" % ((end_time - start_time) / 60.) + ' minutes')

Z_val = 1.96
n_estimates = len(pat_id_arr)
sqrt_n = np.sqrt(n_estimates)

Zdsn = Z_val / sqrt_n

print("detection sensitivity: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(detection_sensitivity_all, Zdsn)
det_sens_perc_av.append(mn)
det_sens_perc_std.append(st)
det_sens_perc_med.append(med)
det_sens_perc_range.append([rmin, rmax])

print("FP rate: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(FP_rate_all, Zdsn)
FP_perc_av.append(mn)
FP_perc_std.append(st)
FP_perc_med.append(med)
FP_perc_range.append([rmin, rmax])

print("PPV: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(PPV_all, Zdsn)
PPV_perc_av.append(mn)
PPV_perc_std.append(st)
PPV_perc_med.append(med)
PPV_perc_range.append([rmin, rmax])

print("F1: ")
mn, st, med, rmin, rmax = af.print_summary_statistics(F1_all, Zdsn)
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


# for the other algorithms, use the function af.save_performance_arrays
# if AIMII_SVM_MODEL:
#     results_array_dir = '/final_result_arrays/'
#     af.save_with_pickle(results_array_dir + 'PPV_original_aimII.pkl', PPV_store)
#     af.save_with_pickle(results_array_dir + 'sensitivity_original_aimII.pkl', sens_store)
#     af.save_with_pickle(results_array_dir + 'F1_original_aimII.pkl', F1_store)
#     af.save_with_pickle(results_array_dir + 'detdelay_original_aimII.pkl', detdelay_store)
#     af.save_with_pickle(results_array_dir + 'FPrate_original_aimII.pkl', FPrate_store)
#     af.save_with_pickle(results_array_dir + 'FPtimes_original_aimII.pkl', FP_times_store)
#     af.save_with_pickle(results_array_dir + 'TP_original_aimII.pkl', TP_store)
#     af.save_with_pickle(results_array_dir + 'FN_original_aimII.pkl', FN_store)
#     af.save_with_pickle(results_array_dir + 'FP_original_aimII.pkl', FP_store)
#     af.save_with_pickle(results_array_dir + 'all_seizures_aimII.pkl', dict_seizures_all)
#     af.save_with_pickle(results_array_dir + 'detected_seizures_original_aimII.pkl', dict_seizures_detected)
# else:
#     results_array_dir = '/final_result_arrays/'
#     af.save_with_pickle(results_array_dir + 'PPV_original.pkl', PPV_store)
#     af.save_with_pickle(results_array_dir + 'sensitivity_original.pkl', sens_store)
#     af.save_with_pickle(results_array_dir + 'F1_original.pkl', F1_store)
#     af.save_with_pickle(results_array_dir + 'detdelay_original.pkl', detdelay_store)
#     af.save_with_pickle(results_array_dir + 'FPrate_original.pkl', FPrate_store)
#     af.save_with_pickle(results_array_dir + 'FPtimes_original.pkl', FP_times_store)
#     af.save_with_pickle(results_array_dir + 'TP_original.pkl', TP_store)
#     af.save_with_pickle(results_array_dir + 'FN_original.pkl', FN_store)
#     af.save_with_pickle(results_array_dir + 'FP_original.pkl', FP_store)
#     af.save_with_pickle(results_array_dir + 'all_seizures.pkl', dict_seizures_all)
#     af.save_with_pickle(results_array_dir + 'detected_seizures_original.pkl', dict_seizures_detected)
