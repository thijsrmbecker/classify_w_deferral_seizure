import sys
sys.path.insert(1, '../')
import numpy as np 
import time
import aux_functions as af

# directories containing the labels and the features
labels_dir_base = '/Annotations_outputs_classifier/'
trustmodels_dir_base = '/trust_models/'
# patient IDs with measurements
pat_id_arr_original = [10,12,13,14,15,16,17,22,24,30,33,34,36,40,41,54,56,59,61,63,64,65,67,68,70,71,72,73,74,75,
                       76,77,78,79,82,83,89,90,93,94,95,99]
pat_id_arr_new = [11, 25, 28, 35, 47, 48, 58, 60, 66, 80, 92, 98]

pat_id_arr = pat_id_arr_original + pat_id_arr_new
n_pat = len(pat_id_arr)

FILTER_TRUST_PERCENTILE = True
RANDOM_TRUST_SCORES = False

dict_predictions = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    fln_predictions_all = labels_dir_base + 'Patient_predictions_rmsa_' + str(pid) + '.npy'
    dict_predictions[pid_str] = af.load_with(fln_predictions_all)
dict_idx_to_check = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    predictions_all = dict_predictions[pid_str]
    idx_to_check = af.calc_idx_to_check_for_seizure(predictions_all, min_n_ones=4)
    dict_idx_to_check[pid_str] = idx_to_check
dict_trust = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    ts_filename = trustmodels_dir_base + 'trust_scores_pat_all_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
    trust_scores_pat = af.load_with(ts_filename)
    # take absolute value, since negative values are also in here
    dict_trust[pid_str] = trust_scores_pat


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

perc_not_trusted_arr = []

# per_arr = [1, 2, 5, 10, 20, 30, 40, 50, 80, 90, 95]
per_arr = [2.0416666]

time_start = time.time()

for per in per_arr: 
    start_time = time.time()
    for cnt in range(n_pat):
        pat_id = pat_id_arr[cnt]
        labels_all = af.load_with(labels_dir_base + 'Patient_labelsaim1_' + str(pat_id) + '.npy')
        pid_str = str(pat_id)
        predictions_all = dict_predictions[pid_str]
        idx_pat_to_check = dict_idx_to_check[pid_str]
        # loop over the recordings of the patient
        TP = 0
        FP = 0
        FN = 0
        total_seizures = 0
        detection_delay_pat = []

        if RANDOM_TRUST_SCORES:
            trust_scores_pat = np.random.rand(predictions_all.shape[0])
        else:
            trust_scores_pat = dict_trust[pid_str]
            # trust scores of SVM model

        seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(label_input=predictions_all, 
                                                            trust_scores=trust_scores_pat, percentile_level=per, idx_to_check=idx_pat_to_check)
        seizure_timings = af.calculate_real_seizure_timings(labels_all)
        TP, FP, FN, total_seizures, det_delay, detected_seizure_timings, fp_times_final = af.precision_metrics_kaat_10s(seizure_flags, seizure_timings)
    
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
            PPV_all.append(PPV)
            F1_all.append(F1)
            detection_sensitivity_all.append(detection_sensitivity)
            det_delay_av = np.mean(detection_delay_pat)
            detection_delay_final.append(det_delay_av)            
        n_hours = float(labels_all.shape[0]) / (60. * 60.)
        FP_rate = 24. * FP / n_hours
        FP_rate_all.append(FP_rate)
    
    end_time = time.time()
    print("calculation time: " + "%.2f" % ((end_time - start_time) / 60.) + ' minutes')

    Z_val = 1.96
    n_estimates = len(pat_id_arr)
    sqrt_n = np.sqrt(n_estimates)

    Zdsn = Z_val / sqrt_n

    print("detection sensitivity: ")
    mn, st, med, rmax, rmin = af.print_summary_statistics(detection_sensitivity_all, Zdsn)
    det_sens_perc_av.append(mn)
    det_sens_perc_std.append(st)
    det_sens_perc_med.append(med)
    det_sens_perc_range.append([rmin, rmax])

    print("FP rate: ")
    mn, st, med, rmax, rmin = af.print_summary_statistics(FP_rate_all, Zdsn)
    FP_perc_av.append(mn)
    FP_perc_std.append(st)
    FP_perc_med.append(med)
    FP_perc_range.append([rmin, rmax])

    print("PPV: ")
    mn, st, med, rmax, rmin = af.print_summary_statistics(PPV_all, Zdsn)
    PPV_perc_av.append(mn)
    PPV_perc_std.append(st)
    PPV_perc_med.append(med)
    PPV_perc_range.append([rmin, rmax])

    print("F1: ")
    mn, st, med, rmax, rmin = af.print_summary_statistics(F1_all, Zdsn)
    F1_perc_av.append(mn)
    F1_perc_std.append(st)
    F1_perc_med.append(med)
    F1_perc_range.append([rmin, rmax])

    print('detection delay: ')
    mn, st, med, rmax, rmin = af.print_summary_statistics(detection_delay_final, Zdsn)
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

print("perc not trusted:")
print(perc_not_trusted_arr)
