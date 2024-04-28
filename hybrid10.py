# CSEC471 Lab 4 - Group 10
# Hybrid system combining, histmean, slidingwindow, and ORB, requiring a majority vote to pass

import os
import random
import ORB_BF
import sliding_windowmulti
import histmean60
from colorama import Fore, Style

def main():
    # Replace these paths with the file paths of your images
    fingerprint = 'fingerprint'
    subject = 'subject'

    f_images = [f for f in os.listdir(fingerprint) if f.endswith('.png')]
    s_images = [f for f in os.listdir(subject) if f.endswith('.png')]

    FRR, FAR, EER = compare_sets(fingerprint, f_images, subject, s_images)
    print("--------------------")
    print("False Reject Rate:", FRR)
    print("False Accept Rate:", FAR)
    print("Equal Error Rate:", EER)
    print("(-■_■)")

    #similarity_scores = []
    #eer_scores = []
    #far_scores = []
    #frr_scores = []

def compare_sets(f_dirt, f_arr, s_dirt, s_arr):
    # Compare fingerprints vars
    total_fingerprints = len(f_arr)
    authenticated_fingerprints = []
    denied_fingerprints = []
    true_auth_counter = 0
    true_deny_counter = 0
    false_auth_counter = 0
    false_reject_counter = 0

    # similarity comparison value determines the strictness of authentication
    # high value = high strictness
    # max = 500 (not recommended)
    orb_threshold = 0.2 #.325
    histmean_threshold = 0.2 #.4
    sliding_threshold = 0.4 #.7

    # compare each index of fingerprints to subject fingerprint to determine authentication:
    for i in range(total_fingerprints):
        print(f'Processing: {i}/{total_fingerprints}')
        # random number added to i, tests for true reject and false positive
        rand = [0, 1]
        rnum = random.choice(rand)
        if i < total_fingerprints-1:
            i2 = i + rnum
        else:
            i2 = i - rnum
        fingerprint_image = os.path.join(f_dirt, f_arr[i])
        subject_image = os.path.join(s_dirt, s_arr[i2])

        #Run each method

        # 1. Sliding
        sliding_similarity_score = sliding_windowmulti.compare_prints(fingerprint_image, subject_image)
        # 2. Orb
        orb_similarity_score = ORB_BF.compare_fingerprints(fingerprint_image, subject_image) / 500
        # 3. HistMean60
        histmean_similarity_score = histmean60.compare_prints(fingerprint_image, subject_image)
        
        #Check for majority
        finalscore = 0
        if sliding_similarity_score[0] > sliding_threshold:
            finalscore += 1
            color = Fore.GREEN
        else:
            color = Fore.RED

        print(f'{color}- Sliding Window Similarity: {sliding_similarity_score[0]}')

        if orb_similarity_score > orb_threshold:
            finalscore += 1
            color = Fore.GREEN
        else:
            color = Fore.RED

        print(f'{color}- ORB Similarity: {orb_similarity_score}')

        if histmean_similarity_score > histmean_threshold:
            finalscore += 1
            color = Fore.GREEN
        else:
            color = Fore.RED
        

        print(f'{color}- histmean60 Similarity: {histmean_similarity_score}')

        #Value can be changed to suit any number of methods, 2 for three methods
        if finalscore >= 2:
            color = Fore.GREEN
            # append tuple to authenticated list
            authenticated_fingerprints.append((fingerprint_image, subject_image))
            # test if true authentication / false authentication
            if fingerprint_image[-11:] == subject_image[-11:]:
                true_auth_counter += 1
            else:
                false_auth_counter += 1
        else:
            color = Fore.RED
            # append tuple to denied list
            denied_fingerprints.append((fingerprint_image, subject_image))
            if fingerprint_image[-11:] == subject_image[-11:]:
                false_reject_counter += 1
            else:
                true_deny_counter += 1
    
        print(f'{color}- Total Similarity: {finalscore}{Style.RESET_ALL}')

    # False Reject Rate
    frr = false_reject_counter / total_fingerprints
    # False Acceptance Rate
    far = false_auth_counter / total_fingerprints
    # EER = Equal Error Rate
    eer = (frr + far) / 2

    print("True Authentication Counter: ",true_auth_counter)
    print("True Deny Counter: ",true_deny_counter)
    print("False Authentication Counter ", false_auth_counter)
    print("False Reject Counter: ", false_reject_counter)

    return frr, far, eer

if __name__ == "__main__":
    main()
