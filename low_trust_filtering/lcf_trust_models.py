import sys
sys.path.insert(1, '../')
import os.path
import numpy as np 
import time
import aux_functions as af

# use threshold strategy (patient-specific percentile) or not (= patient-independent percentile)
THRESHOLD = False
# use the aimII train labels for the trust models or not (= trust models trained on aimII labels)
USE_TRAIN_LABELS = True
# use the SVM model trained on the aimII labels or not (= SVM trained on aimII labels)
AIMII_SVM_MODEL = True
if THRESHOLD:
    str_ext_2 = 'threshold'
else:
    str_ext_2 = 'percentile'
if USE_TRAIN_LABELS:
    str_ext_1 = 'train'
else:
    str_ext_1 = 'neurol'
if AIMII_SVM_MODEL:
    str_extension = str_ext_1 + '_' + str_ext_2 + '_aimII'
else:
    str_extension = str_ext_1 + '_' + str_ext_2


def return_patids_val_folds(fld_number):
    """
    calculate the patient IDs for each fold and run 
    """
    fold_idx = np.arange(6)
    fld_idx_train_val = np.delete(fold_idx, fld_number)
    fld_idx_1 = fld_idx_train_val[0]
    fld_idx_2 = fld_idx_train_val[1]
    fld_idx_3 = fld_idx_train_val[2]
    fld_idx_4 = fld_idx_train_val[3]
    fld_idx_5 = fld_idx_train_val[4]
    fln_pat_ids_1 = 'fold_' + str(fld_idx_1) + '_pat_ids.npy'
    fln_pat_ids_2 = 'fold_' + str(fld_idx_2) + '_pat_ids.npy'
    fln_pat_ids_3 = 'fold_' + str(fld_idx_3) + '_pat_ids.npy'
    fln_pat_ids_4 = 'fold_' + str(fld_idx_4) + '_pat_ids.npy'
    fln_pat_ids_5 = 'fold_' + str(fld_idx_5) + '_pat_ids.npy'
    pat_ids_1 = af.load_with(fln_pat_ids_1)
    pat_ids_2 = af.load_with(fln_pat_ids_2)
    pat_ids_3 = af.load_with(fln_pat_ids_3)
    pat_ids_4 = af.load_with(fln_pat_ids_4)
    pat_ids_5 = af.load_with(fln_pat_ids_5)
    pat_ids_run_1 = np.hstack((pat_ids_1, pat_ids_2))
    pat_ids_run_2 = np.hstack((pat_ids_3, pat_ids_4))
    pat_ids_5_run_1 = pat_ids_5[:5]
    pat_ids_5_run_2 = pat_ids_5[5:]
    pat_ids_run_1 = np.hstack((pat_ids_run_1, pat_ids_5_run_1))
    pat_ids_run_2 = np.hstack((pat_ids_run_2, pat_ids_5_run_2))
    return pat_ids_run_1, pat_ids_run_2


def cross_validate_per(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check, 
                       fld_number, run_number, pat_ids_arr, per_arr_cv):
    """ 
    cross-validate to find the optimal percentile to filter
    we use the F1 score for validation
    """
    n_patients = pat_ids_arr.shape[0]
    per_arr_cv = np.array(per_arr_cv)
    n_per = per_arr_cv.shape[0]
    F1_per = np.zeros((n_per, n_patients))
    for pat_cnt in range(n_patients):
        pat_id = pat_ids_arr[pat_cnt]
        pid_str = str(pat_id)
        labels_all = dict_labels[pid_str]
        predictions_all = dict_predictions[pid_str]
        idx_pat_to_check = dict_idx_to_check[pid_str]
        dict_str = 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pat_id)
        trust_scores_pat = dict_trust_cv[dict_str]
        seizure_timings = dict_seizure_timings[pid_str]
        for per_cnt in range(n_per):
            per = per_arr_cv[per_cnt]
            # loop over the recordings of the patient
            TP = 0
            FP = 0
            FN = 0
            total_seizures = 0
            F1 = 0.
            seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(label_input=predictions_all, 
                                                        trust_scores=trust_scores_pat, percentile_level=per, idx_to_check=idx_pat_to_check)
            # calculate  F1
            TP, FP, FN, total_seizures, det_del, detected_seizure_timings, FP_times = af.precision_metrics_kaat_10s(seizure_flags, seizure_timings)
            if total_seizures > 0: 
                if int(TP) > 0:
                    F1 = 2.*TP / (2.*TP + FP + FN)
                else:
                    F1 = 0.
                F1_per[per_cnt, pat_cnt] = F1 
            else:
                F1_per[per_cnt, pat_cnt] = np.nan
    F1_av = []
    for i in range(n_per):
        F1_av.append(np.nanmean(F1_per[i, :]))
    print('F1 = ' + str(F1_av))
    F1_av = np.array(F1_av)
    where_max = np.argmax(F1_av)
    max_per = per_arr_cv[where_max]
    return max_per


def cross_validate_per_threshold(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check,
                                 fld_number, run_number, pat_ids_arr, per_arr_cv):
    """ 
    cross-validate to find the optimal percentile to filter
    we use the F1 score for validation
    """
    n_patients = pat_ids_arr.shape[0]
    per_arr_cv = np.array(per_arr_cv)
    n_per = per_arr_cv.shape[0]
    F1_per = np.zeros((n_per, n_patients))
    # determine threshold
    all_trust_scores = []
    for pat_id in pat_ids_arr:
        dict_str = 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pat_id)
        all_trust_scores = all_trust_scores + list(dict_trust_cv[dict_str])
    for pat_cnt in range(n_patients):
        pat_id = pat_ids_arr[pat_cnt]
        pid_str = str(pat_id)
        labels_all = dict_labels[pid_str]
        predictions_all = dict_predictions[pid_str]
        idx_pat_to_check = dict_idx_to_check[pid_str]
        dict_str = 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pat_id)
        trust_scores_pat = dict_trust_cv[dict_str]
        seizure_timings = dict_seizure_timings[pid_str]
        for per_cnt in range(n_per):
            per = per_arr_cv[per_cnt]
            threshold_trust_scores = np.percentile(all_trust_scores, per)
            # loop over the recordings of the patient
            TP = 0
            FP = 0
            FN = 0
            total_seizures = 0
            F1 = 0.
            seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_threshold_fast(label_input=predictions_all,
                                                        trust_scores=trust_scores_pat, threshold_value=threshold_trust_scores, idx_to_check=idx_pat_to_check)
            # calculate  F1
            TP, FP, FN, total_seizures, det_del, detected_seizure_timings, FP_times = af.precision_metrics_kaat_10s(seizure_flags, seizure_timings)
            if total_seizures > 0: 
                if int(TP) > 0:
                    F1 = 2.*TP / (2.*TP + FP + FN)
                else:
                    F1 = 0.
                F1_per[per_cnt, pat_cnt] = F1 
            else:
                F1_per[per_cnt, pat_cnt] = np.nan
    F1_av = []
    for i in range(n_per):
        F1_av.append(np.nanmean(F1_per[i, :]))
    print('F1 = ' + str(F1_av))
    F1_av = np.array(F1_av)
    where_max = np.argmax(F1_av)
    max_per = per_arr_cv[where_max]
    max_threshold = np.percentile(all_trust_scores, max_per)
    del all_trust_scores
    return max_threshold


# directories containing the data
labels_dir_base = '/Annotations_outputs_classifier/'
trustmodels_dir_base = '/trust_models/'
if USE_TRAIN_LABELS:
    trustscores_cv_dir_base = '/trust_scores/'
else:
    trustscores_cv_dir_base = '/trust_scores_neur_labels/'
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

per_arr_cv = [0.01, 0.1, 0.5, 1., 2., 5., 10., 20., 40., 80.]
print('# percentages cross-validated on: ')
print('# ' + str(per_arr_cv))
n_percent_cv = len(per_arr_cv)

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
dict_trust_cv = {}
for pid in pat_id_arr:
    for fld_number in range(6):
        for run_number in [0, 1]:
            if AIMII_SVM_MODEL:
                ts_filename = trustscores_cv_dir_base + 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pid) + '_aimII' + '.npy'
            else:
                ts_filename = trustscores_cv_dir_base + 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pid) + '.npy'
            if os.path.isfile(ts_filename):
                trust_scores_pat = af.load_with(ts_filename)
                dict_str = 'fld_' + str(fld_number) + '_run_' + str(run_number) + '_pat_id_' + str(pid)
                dict_trust_cv[dict_str] = trust_scores_pat
dict_trust = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    if AIMII_SVM_MODEL:
        if USE_TRAIN_LABELS:
            ts_filename = trustmodels_dir_base + 'trust_scores_aimII_kaat_pat_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
        else:
            ts_filename = trustmodels_dir_base + 'trust_scores_aimII_pat_all_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
    else:
        if USE_TRAIN_LABELS:
            ts_filename = trustmodels_dir_base + 'trust_scores_kaat_pat_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
        else:
            ts_filename = trustmodels_dir_base + 'trust_scores_pat_all_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
    trust_scores_pat = af.load_with(ts_filename)
    dict_trust[pid_str] = trust_scores_pat
dict_idx_to_check = {}
for pid in pat_id_arr:
    pid_str = str(pid)
    predictions_all = dict_predictions[pid_str]
    idx_to_check = af.calc_idx_to_check_for_seizure(predictions_all, min_n_ones=4)
    dict_idx_to_check[pid_str] = idx_to_check


# arrays to store final performance results
# and detected seizures
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

n_folds = 6
for fld in range(n_folds):
    print("fold: " + str(fld))
    pat_ids_fold_fln = 'fold_' + str(fld) + '_pat_ids.npy'
    pat_ids_fold = af.load_with(pat_ids_fold_fln)
    # patient ids in the two runs for cross-validation
    time_s = time.time()
    pat_ids_run_1, pat_ids_run_2 = return_patids_val_folds(fld_number=fld)
    if THRESHOLD:
        optimal_threshold_1 = cross_validate_per_threshold(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check,
                                fld_number=fld, run_number=0, pat_ids_arr=pat_ids_run_2, per_arr_cv=per_arr_cv)
        optimal_threshold_2 = cross_validate_per_threshold(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check,
                                fld_number=fld, run_number=1, pat_ids_arr=pat_ids_run_1, per_arr_cv=per_arr_cv)
        print("optimal threshold 1: " + str(optimal_threshold_1))
        print("optimal threshold 2: " + str(optimal_threshold_2))
        optimal_av = (optimal_threshold_1 + optimal_threshold_2) / 2.
        print("optimal threshold av: " + str(optimal_av))
    else:
        optimal_per_1 = cross_validate_per(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check, 
                                           fld_number=fld, run_number=0, pat_ids_arr=pat_ids_run_2, per_arr_cv=per_arr_cv)
        optimal_per_2 = cross_validate_per(dict_trust_cv, dict_labels, dict_predictions, dict_seizure_timings, dict_idx_to_check,
                                           fld_number=fld, run_number=1, pat_ids_arr=pat_ids_run_1, per_arr_cv=per_arr_cv)
        print("optimal per 1: " + str(optimal_per_1))
        print("optimal per 2: " + str(optimal_per_2))
        optimal_av = (optimal_per_1 + optimal_per_2) / 2.
        print("optimal per av: " + str(optimal_av))
    time_e = time.time()
    print("calculation time 1 fold: " + "%.2f" % ((time_e - time_s) / 60.)  + ' minutes')
    # calculate metrics for patients in fold
    for pat_id in pat_ids_fold:
        # load trust scores of full model (including all patients except current patient)
        # print('pat_id ' + str(pat_id))
        pid_str = str(pat_id)
        trust_scores_pat = dict_trust[pid_str]
        idx_pat_to_check = dict_idx_to_check[pid_str]
        # Aim II labels of neurologist
        labels_all = dict_labels[pid_str]
        predictions_all = dict_predictions[pid_str]
        # precision metrics
        TP = 0
        FP = 0
        FN = 0
        total_seizures = 0
        detection_delay_pat = []
        # calculate seizure flags 
        if THRESHOLD:
            seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_threshold_fast(label_input=predictions_all, 
                                                        trust_scores=trust_scores_pat, threshold_value=optimal_av, idx_to_check=idx_pat_to_check)
        else:
            seizure_flags, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(label_input=predictions_all, 
                                                            trust_scores=trust_scores_pat, percentile_level=optimal_av, idx_to_check=idx_pat_to_check)
        seizure_timings = dict_seizure_timings[pid_str]
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
            F1_final.append(F1)
            PPV_final.append(PPV)
            detection_sensitivity_final.append(detection_sensitivity)
            detection_delay_av = np.mean(detection_delay_pat)
            detection_delay_final.append(detection_delay_av)
        n_hours = float(labels_all.shape[0]) / (60. * 60.)
        FP_rate = 24. * FP / n_hours
        FP_rate_final.append(FP_rate)

        # store final values in dictionary
        if total_seizures > 0:
            PPV_store[pid_str] = PPV
            sens_store[pid_str] = detection_sensitivity
            F1_store[pid_str] = F1
            detdelay_store[pid_str] = detection_delay_av
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
        FPrate_store[pid_str] = FP_rate
        FP_times_store[pid_str] = FP_times
        dict_seizures_detected[pid_str] = detected_seizure_timings


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

time_end_all = time.time()
print("total calculation time: " + "%.2f" % ((time_end_all - time_start_all) / 60.) + ' minutes')
print('')

# af.save_performance_arrays(str_extension, PPV_store, sens_store, F1_store, detdelay_store, FPrate_store, FP_times_store, TP_store, FN_store, FP_store, dict_seizures_detected)
