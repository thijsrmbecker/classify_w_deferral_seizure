import numpy as np 
import sortednp as snp


def predicted_seizure_segments(start_idx, full_length, interval=300., collate=True):
    """
    Create intervals around predicted seizures (predicted from the model)
    start_idx: start indices of a possible seizure
    full_length: the full length of the EEG
    interval: length (in seconds = number of samples) around the possible seizure
    collate: if True, add segments together if there is overlap
    """
    # number of start indices
    n_s_idx = len(start_idx)
    interval_half = int(interval / 2)
    interval_start_end = np.zeros((n_s_idx, 2))
    final_idx = full_length - 1
    # get start and end times for each interval
    for i in range(n_s_idx):
        interval_start_end[i][0] = start_idx[i] - interval_half
        interval_start_end[i][1] = start_idx[i] + interval_half
    # re-define boundaries that fall out of the segment
    idx_negative = np.where(interval_start_end < 0)
    interval_start_end[idx_negative] = 0
    idx_after_end = np.where(interval_start_end > final_idx)
    interval_start_end[idx_after_end] = final_idx
    # if there is overlap between the segments -> make it one big segment
    i = 0
    while i < (interval_start_end.shape[0]-1):
        if interval_start_end[i, 1] > interval_start_end[i+1, 0]:
            interval_start_end[i, 1] = interval_start_end[i+1, 1]
            interval_start_end = np.delete(interval_start_end, i+1, 0)
        else:
            i+=1
    # if there are less than 'interval' seconds between the segments:
    # remove the segments
    if collate:
        n_segments = interval_start_end.shape[0]
        i = 0
        while i < (interval_start_end.shape[0]-1):
            if interval_start_end[i, 1] > interval_start_end[i+1, 0] - interval:
                interval_start_end[i, 1] = interval_start_end[i+1, 1]
                interval_start_end = np.delete(interval_start_end, i+1, 0)
            else:
                i+=1
        # if the end of the segment is shorter than one interval, include it in the seizure segment
        if interval_start_end[-1, 1] + interval > full_length - 1:
            interval_start_end[-1, 1] = full_length - 1
        # if the segment before the first seizure segment is too short (i.e., it is not a full interval),
        # include it in the seizure segment
        if interval_start_end[0, 0] - interval < 0:
            interval_start_end[0, 0] = 0
    return interval_start_end.astype(int)


def predicted_non_seizure_segments(interval, full_length, seiz_segments):
    """
    Create intervals where no predicted seizures take place
    interval: number of seconds in one interval (= number of samples)
    full_length: full length of the EEG
    seiz_segments: array with [[start_idx, end_idx], [start_idx, end_idx]] of the seizure segments
    """
    seiz_segments = np.array(seiz_segments)
    n_possible_seizure = seiz_segments.shape[0]
    no_seizure_segments = []
    if n_possible_seizure == 0:
        end_idx = full_length - 1
        no_seizure_segments = create_segments_between_start_end_idx(start_idx=0, end_idx=end_idx, interval=interval, full_length=full_length)
        # i = 0
        # final_check = full_length - interval
        # while i < final_check + 1:
        #     no_seizure_segments.append([i, i+interval-1])
        #     i += interval
        # if i != (full_length - interval):
        #     no_seizure_segments[-1][1] = full_length - 1
    else:
        # first do part before first seizure
        # if the seizure starts at zero, it can be included in the main loop
        if seiz_segments[0, 0] != 0:
            start_idx = 0
            end_idx = seiz_segments[0, 0] - 1
            length_seg = end_idx + 1
            extra_segments = create_segments_between_start_end_idx(start_idx=start_idx, end_idx=end_idx, interval=interval, full_length=length_seg)
            no_seizure_segments.extend(extra_segments)
        s_idx_start = 1
        s_idx_end = n_possible_seizure
        for s_idx in range(s_idx_start, s_idx_end):
            start_idx = seiz_segments[s_idx-1, 1] + 1
            end_idx = seiz_segments[s_idx, 0] - 1
            length_seg = end_idx - start_idx + 1
            extra_segments = create_segments_between_start_end_idx(start_idx=start_idx, end_idx=end_idx, interval=interval, full_length=length_seg)  
            no_seizure_segments.extend(extra_segments)   
        # if the last seizure doesn't end at the end of the EEG, also add the last part
        if seiz_segments[-1, 1] != full_length - 1:
            start_idx = seiz_segments[-1, 1] + 1
            end_idx = full_length - 1
            length_seg = end_idx - start_idx + 1
            extra_segments = create_segments_between_start_end_idx(start_idx=start_idx, end_idx=end_idx, interval=interval, full_length=length_seg)  
            no_seizure_segments.extend(extra_segments)   
    return no_seizure_segments


def calc_trust_of_segments_variable_length(trust_arr, segments, lowest_percent=100):
    trust_per_segment = []
    segments = np.array(segments)
    segments = segments.astype(int)
    n_seg = segments.shape[0]
    trust_arr = np.array(trust_arr)
    if n_seg > 0:
        for i in range(n_seg-1):
            idx_first = segments[i, 0]
            idx_final = segments[i, 1] + 1
            trust_seg = trust_arr[idx_first:idx_final]
            length_seg = trust_seg.shape[0]
            length_for_average = int(float(length_seg) * (float(lowest_percent) / float(100)))
            trust_seg_sort = np.sort(trust_seg)
            trust_per_segment.append(np.nanmean(trust_seg_sort[:length_for_average]))
        idx_first = segments[-1, 0]
        if idx_first != trust_arr.shape[0]:
            trust_seg = trust_arr[idx_first:]
            length_seg = (trust_arr.shape[0] - idx_first)
            length_for_average = int(length_seg * (lowest_percent / 100))
            trust_seg_sort = np.sort(trust_seg)
            trust_per_segment.append(np.nanmean(trust_seg_sort[:length_for_average]))
    return trust_per_segment


def collate_trusted_segments(all_segments, non_trusted_segments_idx):
    """
    collate all neighbouring segments, that are not split by an untrustworthy segment
    all_segments: array of the form [[idx_start_1, idx_end_1], [idx_start_2, idx_end_2], ..., [idx_start_n, idx_end_n]]
    non_trust_segments_idx: array with the indices of the untrustworthy segments that need to be removed: [segm_idx_2, segm_idx_5, ...] 
    """
    all_segments = np.array(all_segments)
    n_seg = all_segments.shape[0]
    all_segments_idx = np.arange(n_seg)
    trusted_segments_idx = np.delete(all_segments_idx, non_trusted_segments_idx)
    collated_segments = []
    if trusted_segments_idx.shape[0] == 1:
        collated_segments = np.array([all_segments[trusted_segments_idx[0]]])
    elif trusted_segments_idx.shape[0] == 0:
        collated_segments = np.array([])
    else:
        # if there is more than one trusted segment
        n_seg_trusted = trusted_segments_idx.shape[0]
        final_idx = n_seg_trusted - 1
        new_segment = True
        for i in range(final_idx):
            if new_segment:
                start_idx = all_segments[trusted_segments_idx[i], 0]
                new_segment = False
            end_this_segment = all_segments[trusted_segments_idx[i], 1]
            start_next_segment = all_segments[trusted_segments_idx[i+1], 0]
            if (start_next_segment - end_this_segment) != 1:
                collated_segments.append([start_idx, end_this_segment])
                new_segment = True
        if not new_segment:
            collated_segments.append([start_idx, all_segments[trusted_segments_idx[-1], 1]])
        else:
            collated_segments.append([all_segments[trusted_segments_idx[-1], 0], all_segments[trusted_segments_idx[-1], 1]])
    return collated_segments                
                    

def merge_segments_seizure_no_seizure(seizure_segments, no_seizure_segments):
    seizure_segments = np.array(seizure_segments)
    no_seizure_segments = np.array(no_seizure_segments)
    all_segments = np.concatenate((no_seizure_segments, seizure_segments))
    all_segments = np.sort(all_segments, axis=0)
    all_segments_collated = collate_trusted_segments(all_segments, non_trusted_segments_idx=[])
    return all_segments_collated


def calculate_seizure_flags_w_trustscores_fast(label_input, binary_mask, trust_scores, binary_mask_start_idx_seg):
    """ 
    calculate end time of the seizures detected by the classifier by only taking
    the highest percentile_level trust scores into account
    percentile_level is in [0, 100], gives the lowest n^th percentile
    """
    label_input = label_input.astype(int)
    seizure_flags = []
    n_samples = label_input.shape[0]
    n_less_half_trusted = 0
    high_confidence_points = np.where(binary_mask == 1)[0]
    high_confidence_points = high_confidence_points.astype(int)
    idx_start_segment_check = np.where(binary_mask_start_idx_seg == 1)[0]
    idx_start_segment_check = idx_start_segment_check.astype(int)
    for idx_start_segment in idx_start_segment_check: 
        if idx_start_segment < n_samples - 10:
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
    return np.array(seizure_flags)


def precision_metrics_kaat_10s_w_remove_w_fp(seizure_flags, seizure_timings, start_time_detected_manually, seizure_flag_checked_manually):
    """
    seizure_flags: end of 10-second seizure detection by the algorithm
    seizure_timings: start and end time of real seizures
    The end of the seizure flag from the classifier should fall in the real seizure interval
    false positives are calculated with a 10 second interval between them
    if the seizure is partly in a manual interval, it always counts as a true positive
    if a seizure flag is partly in a manual interval, it is fully checked and doesn't count towards the performance metrics
    """
    TP = 0
    FP = 0
    FN = 0
    det_delay = []
    total_seizure_timings = seizure_timings.shape[0]
    total_nondetected_seizures = total_seizure_timings
    seizure_timings_detected = []
    FP_times = []
    # calculate false and true positives
    seizure_flags_cnt = 0
    for flag_timing in seizure_flags:
        TP_detected = False
        for i in range(total_seizure_timings):  
            if start_time_detected_manually[i] == 1:
                TP_detected = True
            else:
                if (flag_timing >= seizure_timings[i][0]) and (flag_timing <= seizure_timings[i][1]+1):
                    if i not in seizure_timings_detected:
                        TP += 1
                        seizure_timings_detected.append(i)
                        det_delay.append(flag_timing - seizure_timings[i][0])
                    TP_detected = True                            
        if not TP_detected:
            # add false positive only if the seizure is fully in the non-manually checked segment
            if not seizure_flag_checked_manually[seizure_flags_cnt]:
                FP_times.append(flag_timing)
        seizure_flags_cnt += 1
    # calculate false positives
    n_fpt = len(FP_times)
    cnt = 0
    if n_fpt > 0:
        last_fp = FP_times[0]
        FP += 1
        for cnt in range(1, n_fpt):
            fpt = FP_times[cnt]
            if abs(fpt - last_fp) > 10:
                FP += 1
                last_fp = fpt
    # calculate false negatives
    cnt = 0
    for seizure in seizure_timings:
        if start_time_detected_manually[cnt] == 0:
            seizure_detected = False
            for flags in seizure_flags:
                if (flags >= seizure[0]) and (flags <= seizure[1]+9):
                    seizure_detected = True
            if not seizure_detected:
                FN += 1
        cnt += 1
    return TP, FP, FN, total_seizure_timings, det_delay


def seizure_detected_manually(segments, seizure_timings, n_seconds_needed=10):
    """
    check if the seizure is detected in the segment that is manually checked
    """
    segments_collated = collate_trusted_segments(segments, non_trusted_segments_idx=[])
    segments_collated = np.array(segments_collated)
    n_seizures = seizure_timings.shape[0]
    if n_seizures > 0:
        detected_manually_mask = np.zeros((n_seizures, ))
        for i in range(n_seizures):
            seizure_indices = seizure_timings[i]
            for j in range(segments_collated.shape[0]):
                segment_indices = segments_collated[j]
                seizure_indices_arange = np.arange(seizure_indices[0], seizure_indices[1]+1)
                segment_indices_arange = np.arange(segment_indices[0], segment_indices[1]+1)
                if np.intersect1d(seizure_indices_arange, segment_indices_arange).shape[0] >= n_seconds_needed:
                    detected_manually_mask[i] = 1
        detected_manually_mask = detected_manually_mask.astype(int)
    else:
        detected_manually_mask = np.array([]).astype(int)
    return detected_manually_mask


def false_positive_detected_manually(segments, seizure_flags, n_seconds_needed=10):
    """
    check if the false positive is detected in the segment that is manually checked
    """
    segments_collated = collate_trusted_segments(segments, non_trusted_segments_idx=[])
    segments_collated = np.array(segments_collated)
    n_seizures = seizure_flags.shape[0]
    if n_seizures > 0:
        detected_manually_mask = np.zeros((n_seizures, ))
        for i in range(n_seizures):
            seizure_index_start = seizure_flags[i]
            for j in range(segments_collated.shape[0]):
                segment_indices = segments_collated[j]
                seizure_indices_arange = np.arange(seizure_index_start, seizure_index_start+10)
                segment_indices_arange = np.arange(segment_indices[0], segment_indices[1]+1)
                if np.intersect1d(seizure_indices_arange, segment_indices_arange).shape[0] >= n_seconds_needed:
                    detected_manually_mask[i] = 1
        detected_manually_mask = detected_manually_mask.astype(int)
    else:
        detected_manually_mask = np.array([]).astype(int)
    return detected_manually_mask


def seizure_timings_in_segment(segment, seizure_timings, detected_manually_mask):
    """
    check if the seizure is in the current segment checked by the ML algorithm
    segment: one segment of the form [start_idx, end_idx]
    seizure_timings: all seizure timings: [[seiz_start_1, seize_end_1], [seiz_start_2, seize_end_2], ...]
    detected_manually_mask: boolean array of size n_seizure_timings, that stores whether the seizure is partially
    in a segment that is manually checked
    """
    n_seizures = seizure_timings.shape[0]
    if n_seizures > 0:
        seizure_idx_segment = []
        cnt_cutting_error = 0
        for i in range(n_seizures):
            seizure_indices = seizure_timings[i]
            seizure_indices_arange = np.arange(seizure_indices[0], seizure_indices[1]+1)
            segment_indices_arange = np.arange(segment[0], segment[1]+1)
            overlap_seizure_segment = np.intersect1d(seizure_indices_arange, segment_indices_arange).shape[0]
            if overlap_seizure_segment > 0:
                size_seizure = seizure_indices[1] - seizure_indices[0] + 1
                # if there aren't 10 seconds in either the manual and automatic segment, this is a 
                # "cutting error", and the seizure is missed -> it should be stored als a false negative
                if (overlap_seizure_segment < 10) and (size_seizure - overlap_seizure_segment < 10):
                    cnt_cutting_error += 1
                else:
                    seizure_idx_segment.append(i)
        if len(seizure_idx_segment) > 0:
            seizure_in_segment = seizure_timings[seizure_idx_segment]
            detected_manually_in_segment = detected_manually_mask[seizure_idx_segment]
        else:
            seizure_in_segment = []
            detected_manually_in_segment = []
            cnt_cutting_error = 0
    else:
        seizure_in_segment = []
        detected_manually_in_segment = []
        cnt_cutting_error = 0
    return np.array(seizure_in_segment), detected_manually_in_segment, cnt_cutting_error


def seizure_fp_in_segment(segment, seizure_flags, detected_manually_mask):
    """
    check if the false positive (actually seizure flag, but it is used to detect false positives manually)
    is in the current segment checked by the ML algorithm.
    segment: one segment of the form [start_idx, end_idx]
    seizure_flags: all seizure flags: [seiz_flag_1, seiz_flag_2, ...]
    detected_manually_mask: boolean array of size n_seizure_flags, that stores whether the seizure flag is partially
    in a segment that is manually checked. If it is partially in a segment that is manually checked, we assume that
    the neurologist checks the full flag (also the part that is in the automatic checking part)
    """
    n_seizures = seizure_flags.shape[0]
    if n_seizures > 0:
        seizure_idx_segment = []
        for i in range(n_seizures):
            seizure_idx_start = seizure_flags[i]
            seizure_indices_arange = np.arange(seizure_idx_start-9, seizure_idx_start+1)
            segment_indices_arange = np.arange(segment[0], segment[1]+1)
            overlap_seizure_segment = np.intersect1d(seizure_indices_arange, segment_indices_arange).shape[0]
            if overlap_seizure_segment > 0:
                seizure_idx_segment.append(i)
        if len(seizure_idx_segment) > 0:
            seizure_in_segment = seizure_flags[seizure_idx_segment]
            detected_manually_in_segment = detected_manually_mask[seizure_idx_segment]
        else:
            seizure_in_segment = []
            detected_manually_in_segment = []
    else:
        seizure_in_segment = []
        detected_manually_in_segment = []
    return np.array(seizure_in_segment), detected_manually_in_segment


def shift_2d_array(arr, shift_idx, end_idx):
    """
    takes n x 2 array
    and shifts each value by shift_idx
    is used to shift the real seizure timings:
    arr = [[seiz_start_1, seize_end_1], [seiz_start_2, seize_end_2], ...]
    by the starting index of the segment that is checked: segment = [shift_idx, end_idx]
    """
    if arr.shape[0] > 0:
        arr_return = []
        for i in range(arr.shape[0]):
            arr_el_1 = arr[i, 0] 
            arr_el_2 = arr[i, 1]
            if arr_el_1 >= shift_idx and arr_el_2 <= end_idx:
                arr_el_1 = arr_el_1 - shift_idx
                arr_el_2 = arr_el_2 - shift_idx
                arr_return.append([arr_el_1, arr_el_2])
            elif arr_el_1 >= shift_idx and arr_el_2 > end_idx:
                arr_el_1 = arr_el_1 - shift_idx
                arr_return.append([arr_el_1, end_idx])
            elif arr_el_1 < shift_idx and arr_el_2 <= end_idx:
                arr_el_2 = arr_el_2 - shift_idx
                arr_return.append([shift_idx, arr_el_2])
        arr_return = np.array(arr_return)
    else:
        arr_return = np.array([])
    return arr_return


def shift_1d_array(arr, shift_idx, end_idx):
    """
    takes n x 1 array
    and shifts each value by shift_idx
    is used to shift the seizure flags by the start time
    of the segment that is checked segment = [shift_idx, end_idx]
    """
    if arr.shape[0] > 0:
        arr_return = []
        for i in range(arr.shape[0]):
            arr_el = arr[i]
            if arr_el >= shift_idx and arr_el <= end_idx + 9:
                arr_return.append(arr_el - shift_idx)
        arr_return = np.array(arr_return)
    else:
        arr_return = np.array([])
    return arr_return
