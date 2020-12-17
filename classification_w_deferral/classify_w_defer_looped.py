import classify_w_defer_function as cwr

LTF = False

for svm in [True, False]: 
    # use SVM confidences or trust scores
    for rem_only_noseiz in [True]:    
    # remove only the non-seizure segments (i.e., remove all seizure flags at the start) or remove all segments
        for same_per in [True]:   
        # remove same or different percentile per patient
            for p_low in [5]:  # [1, 5, 15, 30, 50, 100]:
            # calculate trust of full segment for lowest p_low percentage of 2-second segments
                print('-----------------------------------------')
                print('svm: ' + str(svm))
                print('rem_only_noseiz: ' + str(rem_only_noseiz))
                print('same_per: ' + str(same_per))
                print('p_low: ' + str(p_low))
                cwr.cwr(SVM_DISTANCES=svm, REMOVE_ONLY_NOSEIZURE=rem_only_noseiz, 
                        SAME_PERCENTILE=same_per, TRUST_SCORE_FILTERING=LTF, p_low_trust=p_low)
