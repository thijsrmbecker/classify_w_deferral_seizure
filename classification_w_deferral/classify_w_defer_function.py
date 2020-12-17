import sys
sys.path.insert(1, '../')
import numpy as np 
import time
import aux_functions as af
import aux_functions_remove as afr
import segment_count as sc


def cwr(SVM_DISTANCES=False, REMOVE_ONLY_NOSEIZURE=True, SAME_PERCENTILE=True, TRUST_SCORE_FILTERING=False, p_low_trust=5):

    time_start_all = time.time()

    #############################################################
    # the indices in the arrays are the real indices in Python  #
    # so idx = [2, 5] should be sliced as ar[2:6]               #
    #############################################################

    # directories containing the labels and the features
    # these are not function for the Github version.
    labels_dir_base = '/Annotations_outputs_classifier/'
    trustmodels_dir_base = '/trust_models/'
    svm_scores_dir = '/SVM_confidence/'
    # patient IDs with measurements
    pat_id_arr_original = [10,12,13,14,15,16,17,22,24,30,33,34,36,40,41,54,56,59,61,63,64,65,67,68,70,71,72,73,74,75,
                        76,77,78,79,82,83,89,90,93,94,95,99]
    pat_id_arr_new = [11, 25, 28, 35, 47, 48, 58, 60, 66, 80, 92, 98]

    pat_id_arr = pat_id_arr_original + pat_id_arr_new
    n_pat = len(pat_id_arr)

    # classify all samples (used for debugging purposes)
    REJECT_NOTHING = False
    # use the SVM model trained on the aimII labels or not (= SVM trained on aimII labels)
    AIMII_SVM_MODEL = True
    # use the trust models trained on the aimII train labels
    TRAIN_LABELS_TRUST = False

    # number of minutes in a segment that contains a possible seizure
    n_minute_interval_seizure = 5
    n_seconds_interval_seizure = n_minute_interval_seizure * 60
    # number of minutes in a segment that doesn't contain a seizure
    n_minute_interval_no_seizure = 5
    n_seconds_interval_no_seizure = n_minute_interval_no_seizure * 60

    # create dictionaries of different inputs
    # done for efficiency purposes (although in the end, influence on speed was small)
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
    for pid in pat_id_arr:
        pid_str = str(pid)
        if AIMII_SVM_MODEL:
            str_predictions = 'Patient_predictions_aimII_rmsa_'
        else:
            str_predictions = 'Patient_predictions_rmsa_'
        fln_predictions_all = labels_dir_base + str_predictions + str(pid) + '.npy'
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
        if SVM_DISTANCES:
            if AIMII_SVM_MODEL:
                scores_svm_dir = '/SVM_confidence_aimII/'
                svm_string = 'svm_aimII_scores_pat_'
            else:
                scores_svm_dir = '/SVM_confidence/'
                svm_string = 'svm_scores_pat_'
            ts_filename = scores_svm_dir + svm_string + str(pid) + '.npy'
            trust_scores_pat = af.load_with(ts_filename)
            # take absolute value
            # dict_trust[pid_str] = abs(trust_scores_pat)
            # take the set-up with the best result for the classify-all scenario:
            # sigmoid with a = 0.01
            a_temp_scaling = 0.1
            dict_trust[pid_str] = af.sigmoid_trust(trust_scores_pat, a=a_temp_scaling)
        else:
            if AIMII_SVM_MODEL:
                if TRAIN_LABELS_TRUST:
                    ts_filename = trustmodels_dir_base + 'trust_scores_aimII_kaat_pat_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'                
                else:
                    ts_filename = trustmodels_dir_base + 'trust_scores_aimII_pat_all_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'                
            else:
                if TRAIN_LABELS_TRUST:
                    ts_filename = trustmodels_dir_base + 'trust_scores_kaat_pat_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
                else:
                    ts_filename = trustmodels_dir_base + 'trust_scores_pat_all_' + str(pid) + '_alpha_0p05_k_10_filtering_none_noseizesegments_100pca_20.npy'
            trust_scores_pat = af.load_with(ts_filename)
            dict_trust[pid_str] = trust_scores_pat
    dict_seizure_flags = {}
    for pid in pat_id_arr:
        pid_str = str(pid)
        start_idx = dict_idx_to_check[pid_str]
        predictions_all = dict_predictions[pid_str]
        if TRUST_SCORE_FILTERING:
            trust_scores_pat = dict_trust[pid_str]
            trust_scores_pat = np.array(trust_scores_pat)
            start_idx = np.array(start_idx)
            start_idx = start_idx.astype(int)
            # if SVM_DISTANCES:
            #     seizure_flags_all, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(
            #                 label_input=predictions_all, trust_scores=trust_scores_pat, percentile_level=0.1, idx_to_check=start_idx)
            # else:
            if AIMII_SVM_MODEL:
                seizure_flags_all, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(
                            label_input=predictions_all, trust_scores=trust_scores_pat, percentile_level=9.6, idx_to_check=start_idx)
            else:
                seizure_flags_all, n_not_trusted = af.calculate_seizure_flags_w_trustscores_fast(
                            label_input=predictions_all, trust_scores=trust_scores_pat, percentile_level=2.0416666, idx_to_check=start_idx)
        else:
            seizure_flags_all = af.calculate_seizure_flags_fast(predictions_all, start_idx)
        dict_seizure_flags[pid_str] = seizure_flags_all 

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

    # number of seizures per patient
    n_seizures = [0, 1, 8, 1, 1, 8, 2, 2, 1, 0, 3, 0, 3, 2, 1, 0, 1, 1, 1, 2, 0, 5, 1, 4, 17, 5, 3, 2, 
                3, 2, 6, 0, 5, 2, 6, 2, 0, 1, 2, 3, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print("total number of seizures = " + str(sum(n_seizures))) 

    perc_removed_arr = []
    per_arr = [0, 0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]
    # if the segments are removed with different percentiles
    # calculate the distribution of all trust scores
    if not SAME_PERCENTILE:
        all_trust_all_patient = []
        for cnt in range(n_pat):
            pat_id = pat_id_arr[cnt]
            pid_str = str(pat_id)
            predictions_all = dict_predictions[pid_str]
            labels_all = dict_labels[pid_str]
            seizure_flags_all = dict_seizure_flags[pid_str]
            # put these into "n_minute_interval"-minute seizure segments
            length_full_eeg = predictions_all.shape[0]
            # for each seizure flag -> take -2.5 minutes and +2.5 minutes
            # [s_1, e_1], [s_2, e_2], ...
            # if there is overlap overlap e_i and s_i+1 -> merge 
            if len(seizure_flags_all) > 0:
                possible_seizure_segments = afr.predicted_seizure_segments(seizure_flags_all, full_length=length_full_eeg, interval=n_seconds_interval_seizure, collate=True)
            else:
                possible_seizure_segments = np.array([])
            # create non-seizure segments
            non_seizure_segments = afr.predicted_non_seizure_segments(interval=n_seconds_interval_no_seizure, full_length=length_full_eeg, seiz_segments=possible_seizure_segments)  
            # if there are no "non-seizure segments": print a warning
            # this should normally only happen if > 99% of segments are removed
            if np.array(non_seizure_segments).shape[0] == 0:
                print("did not find any \"no seizure segments\" for patient id: " + str(pat_id))
                print("")
            trust_scores_pat = dict_trust[pid_str] 
            # has to be casted as "float32" to avoid errors with "nanmean". Is an issue with numpy, not the code here.
            trust_scores_pat = np.array(trust_scores_pat, dtype='float32')
            nonseizure_segments_trusts = afr.calc_trust_of_segments_variable_length(trust_arr=trust_scores_pat, segments=non_seizure_segments, lowest_percent=p_low_trust) 
            nonseizure_segments_trusts = np.array(nonseizure_segments_trusts)
            if REMOVE_ONLY_NOSEIZURE:
                all_trust_all_patient = all_trust_all_patient + list(nonseizure_segments_trusts)
            else:
                if possible_seizure_segments.shape[0] > 0:
                    possible_seizure_segments_trusts = afr.calc_trust_of_segments_variable_length(trust_arr=trust_scores_pat, segments=possible_seizure_segments, lowest_percent=p_low_trust)
                    all_trust_all_patient = all_trust_all_patient + list(nonseizure_segments_trusts) 
                    all_trust_all_patient = all_trust_all_patient + list(possible_seizure_segments_trusts)
                else:
                    all_trust_all_patient = all_trust_all_patient + list(nonseizure_segments_trusts)
        # impute NaNs by the median trust score
        all_trust_all_patient = np.array(all_trust_all_patient)
        nan_mask = np.isnan(all_trust_all_patient)
        all_trust_all_patient[nan_mask] = np.nanmedian(all_trust_all_patient)

    mean_seg_length = [] 
    std_seg_length = []
    n_seg_per_day = []

    # start of the main loop 
    for per in per_arr:
        print("")
        print("per = " + str(per))
        perc_checked = 0.
        FP_tot = 0   # for debugging purposes

        # define arrays that contain the performances
        F1_final = []
        PPV_final = []
        detection_sensitivity_final = []
        FP_rate_final = []
        detection_delay_final = []

        # determine threshold for patient-specific removal strategy
        if not SAME_PERCENTILE:
            threshold_per = np.percentile(all_trust_all_patient, per)
        
        mean_length_pat_arr = []  
        std_length_pat_arr = []
        n_seg_pat_arr = []

        for cnt in range(n_pat):
            pat_id = pat_id_arr[cnt]
            pid_str = str(pat_id)
            ########################################################
            # find all possible segments containing a seizure flag #
            ########################################################
            predictions_all = dict_predictions[pid_str]
            labels_all = dict_labels[pid_str]
            full_seizure_timings = dict_seizure_timings[pid_str]
            seizure_flags_all = dict_seizure_flags[pid_str]
            length_full_eeg = predictions_all.shape[0]
            # for each seizure flag -> take - (n_minute_interval / 2) minutes and + (n_minute_interval / 2) minutes
            # [s_1, e_1], [s_2, e_2], ...
            # if there is overlap overlap e_i and s_i+1 -> merge 
            if len(seizure_flags_all) > 0:
                possible_seizure_segments = afr.predicted_seizure_segments(seizure_flags_all, full_length=length_full_eeg, interval=n_seconds_interval_seizure, collate=True)
            else:
                possible_seizure_segments = np.array([])
            # create non-seizure segments
            non_seizure_segments = afr.predicted_non_seizure_segments(interval=n_seconds_interval_no_seizure, full_length=length_full_eeg, seiz_segments=possible_seizure_segments)  
            # if there are no "non-seizure segments": print a warning
            # this should normally only happen if > 99% of segments are removed
            if np.array(non_seizure_segments).shape[0] == 0:
                print("did not find any \"no seizure segments\" for patient id: " + str(pat_id))
                print("")
            #################################################
            # calculate average trust score of each segment #
            #################################################
            trust_scores_pat = dict_trust[pid_str] 
            # has to be casted as "float32" to avoid errors with "nanmean". This is an issue with numpy, not this code.
            trust_scores_pat = np.array(trust_scores_pat, dtype='float32')
            nonseizure_segments_trusts = afr.calc_trust_of_segments_variable_length(trust_arr=trust_scores_pat, segments=non_seizure_segments, lowest_percent=p_low_trust)
            nonseizure_segments_trusts = np.array(nonseizure_segments_trusts)
            if possible_seizure_segments.shape[0] > 0:
                possible_seizure_segments_trusts = afr.calc_trust_of_segments_variable_length(trust_arr=trust_scores_pat, segments=possible_seizure_segments, lowest_percent=p_low_trust)
                possible_seizure_segments_trusts = np.array(possible_seizure_segments_trusts)
                # combine all the trust scores for the seizure / nonseizure segments
                if not REMOVE_ONLY_NOSEIZURE:
                    all_segments_trusts = list(nonseizure_segments_trusts) + list(possible_seizure_segments_trusts)
                else:
                    all_segments_trusts = list(nonseizure_segments_trusts)
                # impute NaNs with median
                all_segments_trusts = np.array(all_segments_trusts)
                nan_mask = np.isnan(all_segments_trusts)
                nanmed_all = np.nanmedian(all_segments_trusts)
                all_segments_trusts[nan_mask] = nanmed_all
                nan_mask = np.isnan(possible_seizure_segments_trusts)
                possible_seizure_segments_trusts[nan_mask] = nanmed_all
                nan_mask = np.isnan(nonseizure_segments_trusts)
                nonseizure_segments_trusts[nan_mask] = nanmed_all
                if SAME_PERCENTILE:
                    low_confidence_segments_seizure = np.where(possible_seizure_segments_trusts <= np.percentile(all_segments_trusts, per))[0]
                else:
                    low_confidence_segments_seizure = np.where(possible_seizure_segments_trusts <= threshold_per)[0]
                if low_confidence_segments_seizure.shape[0] == 0:
                    low_confidence_segments_seizure = np.array([])
                if REJECT_NOTHING:
                    low_confidence_segments_seizure = np.array([])
                low_confidence_segments_seizure = np.array(low_confidence_segments_seizure)
                possible_seizure_segments = np.array(possible_seizure_segments)
                if low_confidence_segments_seizure.shape[0] > 0:
                    low_confidence_segments_full = possible_seizure_segments[low_confidence_segments_seizure]
                else:
                    low_confidence_segments_full = np.array([])
                if REMOVE_ONLY_NOSEIZURE:
                    low_confidence_segments_full = possible_seizure_segments
                    segments_to_check_seizure = []
                else:
                    segments_to_check_seizure = afr.collate_trusted_segments(all_segments=possible_seizure_segments, non_trusted_segments_idx=low_confidence_segments_seizure)
                manual_detection_seizure_segments = afr.seizure_detected_manually(low_confidence_segments_full, full_seizure_timings, n_seconds_needed=10)
                manual_detection_flags_seizure_segments = afr.false_positive_detected_manually(low_confidence_segments_full, seizure_flags_all, n_seconds_needed=1)
            else:
                manual_detection_seizure_segments = np.array([])
                manual_detection_flags_seizure_segments = np.array([])
                segments_to_check_seizure = []
                all_segments_trusts = nonseizure_segments_trusts 
                nan_mask = np.isnan(all_segments_trusts)
                all_segments_trusts[nan_mask] = np.nanmedian(all_segments_trusts)
                nan_mask = np.isnan(nonseizure_segments_trusts)
                nonseizure_segments_trusts[nan_mask] = np.nanmedian(all_segments_trusts)
            #########################################################
            # defer the low-trusted segments, and classify the rest #
            #########################################################
            if SAME_PERCENTILE:   # remove separate percentile per patient
                low_confidence_segments = np.where(nonseizure_segments_trusts <= np.percentile(all_segments_trusts, per))[0]
            else:
                low_confidence_segments = np.where(nonseizure_segments_trusts <= threshold_per)[0]
            if REJECT_NOTHING:
                low_confidence_segments = np.array([])
            # calculate segments to check
            segments_to_check_non_seizure = afr.collate_trusted_segments(all_segments=non_seizure_segments, non_trusted_segments_idx=low_confidence_segments)
            low_confidence_segments = np.array(low_confidence_segments)
            non_seizure_segments = np.array(non_seizure_segments)
            low_confidence_segments_full = non_seizure_segments[low_confidence_segments]
            manual_detection_nonseizure_segments = afr.seizure_detected_manually(low_confidence_segments_full, full_seizure_timings, n_seconds_needed=10)
            manual_detection_flags_nonseizure_segments = afr.false_positive_detected_manually(low_confidence_segments_full, seizure_flags_all, n_seconds_needed=1)
            if manual_detection_seizure_segments.shape[0] > 0:
                manual_detection_segments = manual_detection_nonseizure_segments | manual_detection_seizure_segments
            else:
                manual_detection_segments = manual_detection_nonseizure_segments
            if manual_detection_flags_seizure_segments.shape[0] > 0:
                manual_detection_flags_segments = manual_detection_flags_seizure_segments | manual_detection_flags_nonseizure_segments
            else:
                manual_detection_flags_segments = manual_detection_flags_nonseizure_segments
            if possible_seizure_segments.shape[0] > 0:
                segments_to_check_seizure = np.array(segments_to_check_seizure)
                segments_to_check_non_seizure = np.array(segments_to_check_non_seizure)
                if (segments_to_check_seizure.shape[0] > 0) and (segments_to_check_non_seizure.shape[0] > 0):
                    segments_to_check = afr.merge_segments_seizure_no_seizure(seizure_segments=segments_to_check_seizure, no_seizure_segments=segments_to_check_non_seizure)
                elif segments_to_check_non_seizure.shape[0] > 0:
                    segments_to_check = segments_to_check_non_seizure
                else:
                    segments_to_check = segments_to_check_seizure
            else:
                segments_to_check = segments_to_check_non_seizure
            n_hours_checked = 0
            segm_cnt = 0
            TP = 0
            FP = 0
            FN = 0
            total_seizures = 0
            n_seconds_checked = 0.
            n_seizures_in_non_rejected_part = 0
            detection_delay_segments = []
            n_seg_pat, mean_length_pat, std_length_pat = sc.count_segments(segments_to_check, total_length=length_full_eeg)
            mean_length_pat_arr.append(mean_length_pat)
            std_length_pat_arr.append(std_length_pat)
            n_seg_pat_arr.append(n_seg_pat)
            for segment in segments_to_check:
                seizure_in_segment, detected_manually_in_segment, cnt_cutting_error = afr.seizure_timings_in_segment(segment, full_seizure_timings, manual_detection_segments)
                predicted_in_segment, predicted_manually_in_segment = afr.seizure_fp_in_segment(segment, seizure_flags_all, manual_detection_flags_segments)
                start_idx = int(segment[0])
                end_idx = int(segment[1] + 1)
                n_seconds_checked += (end_idx - start_idx)
                seizure_timings = afr.shift_2d_array(seizure_in_segment, start_idx, end_idx)
                seizure_flags = afr.shift_1d_array(seizure_flags_all, start_idx, end_idx)
                TP_seg, FP_seg, FN_seg, total_seizures_seg, det_delay = afr.precision_metrics_kaat_10s_w_remove_w_fp(seizure_flags, seizure_timings, detected_manually_in_segment, predicted_manually_in_segment)
                detection_delay_segments = detection_delay_segments + det_delay
                TP += TP_seg
                FP += FP_seg
                FN += FN_seg
                FN += cnt_cutting_error
                total_seizures += (total_seizures_seg - sum(detected_manually_in_segment))
                n_seizures_in_non_rejected_part += seizure_timings.shape[0]
            perc_checked += (float(n_seconds_checked) / float(length_full_eeg))
            if REMOVE_ONLY_NOSEIZURE:
                if n_seizures[cnt] > 0:
                    seizures_detected_by_neurologist = n_seizures[cnt] - total_seizures
                    TP = float(seizures_detected_by_neurologist)
                    FN = float(total_seizures)
                    if seizures_detected_by_neurologist > 0:
                        detection_sensitivity = TP / (TP + FN)
                        F1 = 2.*TP / (2.*TP + FN)
                        PPV = 1.
                    else:
                        detection_sensitivity = 0.
                        F1 = 0.  
                        PPV = 0.              
                    F1_final.append(F1)
                    PPV_final.append(PPV)
                    detection_sensitivity_final.append(detection_sensitivity)
                detection_delay_final.append(0)
                FP_rate_final.append(0)
            else:
                if n_seizures[cnt] > 0: 
                    seizures_detected_by_neurologist = n_seizures[cnt] - total_seizures
                    TP = float(TP) + float(seizures_detected_by_neurologist)
                    if int(TP) > 0:
                        detection_sensitivity = TP / (TP + FN)
                        PPV = TP / (TP + FP)
                        F1 = 2.*TP / (2.*TP + FP + FN)
                    else:
                        detection_sensitivity = 0.
                        F1 = 0.
                        PPV = 0.
                    PPV_final.append(PPV)
                    F1_final.append(F1)
                    PPV_final.append(PPV)
                    detection_sensitivity_final.append(detection_sensitivity)
                    det_delay_neurologist = [0] * seizures_detected_by_neurologist
                    det_delay_combined = detection_delay_segments + det_delay_neurologist
                    detection_delay_final.append(np.mean(det_delay_combined))
                n_hours = float(labels_all.shape[0]) / (60. * 60.)
                n_hours_checked = float(n_seconds_checked) / (60. * 60.)
                if int(n_hours_checked) == 0:
                    FP_rate = 0
                else:
                    FP_rate = 24. * FP / n_hours_checked
                FP_rate_final.append(FP_rate)

            FP_tot += FP
        perc_removed_arr.append(1. - (perc_checked / float(n_pat)))

        mean_seg_length.append(np.mean(np.array(mean_length_pat_arr)))
        std_seg_length.append(np.mean(np.array(std_length_pat_arr)))
        n_seg_per_day.append(np.mean(np.array(n_seg_pat_arr)))

        Z_val = 1.96
        n_estimates = len(pat_id_arr)
        sqrt_n = np.sqrt(n_estimates)
        Zdsn = Z_val / sqrt_n

        print("FP rate: ")
        mn, st, med, rmax, rmin = af.print_summary_statistics(FP_rate_final, Zdsn)
        FP_perc_av.append(mn)
        FP_perc_std.append(st)
        FP_perc_med.append(med)
        FP_perc_range.append([rmin, rmax])
        
        print("detection sensitivity: ")
        mn, st, med, rmax, rmin = af.print_summary_statistics(detection_sensitivity_final, Zdsn)
        det_sens_perc_av.append(mn)
        det_sens_perc_std.append(st)
        det_sens_perc_med.append(med)
        det_sens_perc_range.append([rmin, rmax])

        print("PPV: ")
        mn, st, med, rmax, rmin = af.print_summary_statistics(PPV_final, Zdsn)
        PPV_perc_av.append(mn)
        PPV_perc_std.append(st)
        PPV_perc_med.append(med)
        PPV_perc_range.append([rmin, rmax])

        print("F1: ")
        mn, st, med, rmax, rmin = af.print_summary_statistics(F1_final, Zdsn)
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
        
    # np.set_printoptions(precision=3)
    # print("# detection sensitivity: ")
    # print('sens = ' + str(np.array(det_sens_perc_av)))
    # print('sens_std = ' + str(np.array(det_sens_perc_std)))
    # print('sens_med = ' + str(np.array(det_sens_perc_med)))
    # print('sens_range = ' + str(np.array(det_sens_perc_range)))

    # print("# FP rate: ")
    # print('FP = ' + str(np.array(FP_perc_av)))
    # print('FP_std = ' + str(np.array(FP_perc_std)))
    # print('FP_med = ' + str(np.array(FP_perc_med)))
    # print('FP_range = ' + str(np.array(FP_perc_range)))

    # print("# PPV: ")
    # print('PPV = ' + str(np.array(PPV_perc_av)))
    # print('PPV_std = ' + str(np.array(PPV_perc_std)))
    # print('PPV_med = ' + str(np.array(PPV_perc_med)))
    # print('PPV_range = ' + str(np.array(PPV_perc_range)))

    # print("# F1: ")
    # print('F1 = ' + str(np.array(F1_perc_av)))
    # print('F1_std = ' + str(np.array(F1_perc_std)))
    # print('F1_med = ' + str(np.array(F1_perc_med)))
    # print('F1_range = ' + str(np.array(F1_perc_range)))

    # print("# percent removed: ")
    # print('perc_removed = ' + str(np.array(perc_removed_arr)))

    # print("# detection delay: ")
    # print('dd = ' + str(np.array(dd_perc_av)))
    # print('dd_std = ' + str(np.array(dd_perc_std)))
    # print('dd_med = ' + str(np.array(dd_perc_med)))
    # print('dd_range = ' + str(np.array(dd_perc_range)))

    # add the 0% performance to the array
    perc_removed_arr = [0] + list(perc_removed_arr)
    if AIMII_SVM_MODEL:
        if not TRUST_SCORE_FILTERING:
            # sensitivity
            det_sens_perc_av = [0.830] + list(det_sens_perc_av)
            det_sens_perc_std = [0.300] + list(det_sens_perc_std)
            det_sens_perc_med = [1] + list(det_sens_perc_med)
            det_sens_perc_range = [[0, 1]] + list(det_sens_perc_range)
            # FP rate
            FP_perc_av = [17.2] + list(FP_perc_av)
            FP_perc_std = [21.0] + list(FP_perc_std)
            FP_perc_med = [9.54] + list(FP_perc_med)
            FP_perc_range = [[0.26, 98.2]] + list(FP_perc_range)
            # PPV
            PPV_perc_av = [0.126] + list(PPV_perc_av)
            PPV_perc_std = [0.174] + list(PPV_perc_std)
            PPV_perc_med = [0.055555] + list(PPV_perc_med)
            PPV_perc_range = [[0, 0.875]] + list(PPV_perc_range)
            # F1
            F1_perc_av = [0.183] + list(F1_perc_av)
            F1_perc_std = [0.203] + list(F1_perc_std)
            F1_perc_med = [0.105] + list(F1_perc_med)
            F1_perc_range = [[0, 0.875]] + list(F1_perc_range)
            # detection delay
            dd_perc_av = [21.6] + list(dd_perc_av)
            dd_perc_std = [17.4] + list(dd_perc_std)
            dd_perc_med = [16.875] + list(dd_perc_med)
            dd_perc_range = [[1, 89]] + list(dd_perc_range)
        else:
            # sensitivity
            det_sens_perc_av = [0.819] + list(det_sens_perc_av)
            det_sens_perc_std = [0.325] + list(det_sens_perc_std)
            det_sens_perc_med = [1] + list(det_sens_perc_med)
            det_sens_perc_range = [[0, 1]] + list(det_sens_perc_range)
            # FP rate
            FP_perc_av = [10.6] + list(FP_perc_av)
            FP_perc_std = [14.6] + list(FP_perc_std)
            FP_perc_med = [5.3] + list(FP_perc_med)
            FP_perc_range = [[0, 75.8]] + list(FP_perc_range)
            # PPV
            PPV_perc_av = [0.203] + list(PPV_perc_av)
            PPV_perc_std = [0.239] + list(PPV_perc_std)
            PPV_perc_med = [0.125] + list(PPV_perc_med)
            PPV_perc_range = [[0, 1]] + list(PPV_perc_range)
            # F1
            F1_perc_av = [0.277] + list(F1_perc_av)
            F1_perc_std = [0.271] + list(F1_perc_std)
            F1_perc_med = [0.166666] + list(F1_perc_med)
            F1_perc_range = [[0, 1]] + list(F1_perc_range)
            # detection delay
            dd_perc_av = [22.2] + list(dd_perc_av)
            dd_perc_std = [18.0] + list(dd_perc_std)
            dd_perc_med = [16.4] + list(dd_perc_med)
            dd_perc_range = [[2, 90]] + list(dd_perc_range)
    else:
        if TRUST_SCORE_FILTERING:
            # sensitivity
            det_sens_perc_av = [0.704] + list(det_sens_perc_av)
            det_sens_perc_std = [0.387] + list(det_sens_perc_std)
            det_sens_perc_med = [1] + list(det_sens_perc_med)
            det_sens_perc_range = [[0, 1]] + list(det_sens_perc_range)
            # FP rate
            FP_perc_av = [2.2] + list(FP_perc_av)
            FP_perc_std = [4.46] + list(FP_perc_std)
            FP_perc_med = [0.576] + list(FP_perc_med)
            FP_perc_range = [[0, 23.6]] + list(FP_perc_range)
            # PPV
            PPV_perc_av = [0.504] + list(PPV_perc_av)
            PPV_perc_std = [0.408] + list(PPV_perc_std)
            PPV_perc_med = [0.3333] + list(PPV_perc_med)
            PPV_perc_range = [[0, 1]] + list(PPV_perc_range)
            # F1
            F1_perc_av = [0.517] + list(F1_perc_av)
            F1_perc_std = [0.370] + list(F1_perc_std)
            F1_perc_med = [0.5] + list(F1_perc_med)
            F1_perc_range = [[0, 1]] + list(F1_perc_range)
            # detection delay
            dd_perc_av = [21.3] + list(dd_perc_av)
            dd_perc_std = [11.9] + list(dd_perc_std)
            dd_perc_med = [19.7] + list(dd_perc_med)
            dd_perc_range = [[3, 56]] + list(dd_perc_range)
        else:
            # sensitivity
            det_sens_perc_av = [0.641] + list(det_sens_perc_av)
            det_sens_perc_std = [0.415] + list(det_sens_perc_std)
            det_sens_perc_med = [1] + list(det_sens_perc_med)
            det_sens_perc_range = [[0, 1]] + list(det_sens_perc_range)
            # FP rate
            FP_perc_av = [2.9] + list(FP_perc_av)
            FP_perc_std = [5.6] + list(FP_perc_std)
            FP_perc_med = [1.2] + list(FP_perc_med)
            FP_perc_range = [[0, 31.5]] + list(FP_perc_range)
            # PPV
            PPV_perc_av = [0.389] + list(PPV_perc_av)
            PPV_perc_std = [0.389] + list(PPV_perc_std)
            PPV_perc_med = [0.231] + list(PPV_perc_med)
            PPV_perc_range = [[0, 1]] + list(PPV_perc_range)
            # F1
            F1_perc_av = [0.397] + list(F1_perc_av)
            F1_perc_std = [0.342] + list(F1_perc_std)
            F1_perc_med = [0.316] + list(F1_perc_med)
            F1_perc_range = [[0, 1]] + list(F1_perc_range)
            # detection delay
            dd_perc_av = [22.1] + list(dd_perc_av)
            dd_perc_std = [13.2] + list(dd_perc_std)
            dd_perc_med = [19.3] + list(dd_perc_med)
            dd_perc_range = [[2, 55]] + list(dd_perc_range)


    final_array_results = '/result_arrays/'

    if SVM_DISTANCES:
        str_save_1 = '_svm_'
    else:
        str_save_1 = '_trust_'
    if REMOVE_ONLY_NOSEIZURE:
        str_save_2 = 'noseize_'
    else:
        str_save_2 = 'all_'
    if SAME_PERCENTILE:
        str_save_3 = 'perc_'
    else:
        str_save_3 = 'thres_'
    if AIMII_SVM_MODEL:
        str_save_4 = str(p_low_trust) + '_aimII'
    else:
        str_save_4 = str(p_low_trust)  # + '_' + str(a_temp_scaling)
    if TRAIN_LABELS_TRUST:
        str_save_4 = str_save_4 + '_trust_aimIItrain'
    str_save_5 = '_wtrust'


    # uncomment this for counting the number and average length of the deferred segments
    perc_removed_str = 'perc_removed_nseg_' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
    # seg_length mean str
    seg_mean_str = 'seg' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
    # seg_length std str
    seg_std_str = 'seg_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
    # n_seg string
    seg_number_str = 'nseg_' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
    af.save_with(final_array_results + perc_removed_str, perc_removed_arr)
    af.save_with(final_array_results + seg_mean_str, mean_seg_length)
    af.save_with(final_array_results + seg_std_str, std_seg_length)
    af.save_with(final_array_results + seg_number_str, n_seg_per_day)


    if TRUST_SCORE_FILTERING:
        perc_removed_str = 'perc_removed' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        # mean str
        sens_mean_str = 'sens' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fprate_mean_str = 'FPrate' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fone_mean_str = 'F1' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        ppv_mean_str = 'PPV' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        det_delay_mean_str = 'ddelay' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        # std str
        sens_std_str = 'sens_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fprate_std_str = 'FPrate_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fone_std_str = 'F1_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        ppv_std_str = 'PPV_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        det_delay_std_str = 'ddelay_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        # median str
        sens_med_str = 'sens_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fprate_med_str = 'FPrate_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fone_med_str = 'F1_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        ppv_med_str = 'PPV_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        det_delay_med_str = 'ddelay_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        # range str
        sens_range_str = 'sens_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fprate_range_str = 'FPrate_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        fone_range_str = 'F1_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        ppv_range_str = 'PPV_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
        det_delay_range_str = 'ddelay_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + str_save_5 + '.npy'
    else:
        perc_removed_str = 'perc_removed' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        # mean str
        sens_mean_str = 'sens' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fprate_mean_str = 'FPrate' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fone_mean_str = 'F1' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        ppv_mean_str = 'PPV' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        det_delay_mean_str = 'ddelay' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        # std str
        sens_std_str = 'sens_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fprate_std_str = 'FPrate_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fone_std_str = 'F1_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        ppv_std_str = 'PPV_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        det_delay_std_str = 'ddelay_std' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        # median str
        sens_med_str = 'sens_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fprate_med_str = 'FPrate_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fone_med_str = 'F1_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        ppv_med_str = 'PPV_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        det_delay_med_str = 'ddelay_med' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        # range str
        sens_range_str = 'sens_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fprate_range_str = 'FPrate_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        fone_range_str = 'F1_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        ppv_range_str = 'PPV_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'
        det_delay_range_str = 'ddelay_range' + str_save_1 + str_save_2 + str_save_3 + str_save_4 + '.npy'

    # CHANGE
    # af.save_with(final_array_results + perc_removed_str, perc_removed_arr)
    # # mean arrays
    # af.save_with(final_array_results +  sens_mean_str, det_sens_perc_av)
    # af.save_with(final_array_results + fprate_mean_str, FP_perc_av)
    # af.save_with(final_array_results +  fone_mean_str, F1_perc_av)
    # af.save_with(final_array_results + ppv_mean_str, PPV_perc_av)
    # af.save_with(final_array_results + det_delay_mean_str, dd_perc_av)
    # # std arrays
    # af.save_with(final_array_results + sens_std_str, det_sens_perc_std)
    # af.save_with(final_array_results + fprate_std_str, FP_perc_std)
    # af.save_with(final_array_results + fone_std_str, F1_perc_std)
    # af.save_with(final_array_results + ppv_std_str, PPV_perc_std)
    # af.save_with(final_array_results + det_delay_std_str, dd_perc_std)
    # # median arrays
    # af.save_with(final_array_results + sens_med_str, det_sens_perc_med)
    # af.save_with(final_array_results + fprate_med_str, FP_perc_med)
    # af.save_with(final_array_results + fone_med_str, F1_perc_med)
    # af.save_with(final_array_results + ppv_med_str, PPV_perc_med)
    # af.save_with(final_array_results + det_delay_med_str, dd_perc_med)
    # # range arrays
    # af.save_with(final_array_results + sens_range_str, det_sens_perc_range)
    # af.save_with(final_array_results + fprate_range_str, FP_perc_range)
    # af.save_with(final_array_results + fone_range_str, F1_perc_range)
    # af.save_with(final_array_results + ppv_range_str, PPV_perc_range)
    # af.save_with(final_array_results + det_delay_range_str, dd_perc_range)

    time_end_all = time.time()
    print("total calculation time: " + "%.2f" % ((time_end_all - time_start_all) / 60.) + ' minutes')
    print('')
