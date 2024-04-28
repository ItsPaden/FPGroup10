# Ben Tinsley : CSEC 472 - Lab 4
# ORB method of comparison

import cv2
import os
import random


def compare_fingerprints(f_path, s_path):
    # loading fingerprint images
    f_image = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
    s_image = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)

    # ORB detector for feature extraction
    orb = cv2.ORB_create()
    keypoint1, descriptors1 = orb.detectAndCompute(f_image, None)
    keypoint2, descriptors2 = orb.detectAndCompute(s_image, None)

    # Brute Force feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    # similarity score = number of matches in descriptors
    similarity_score = len(matches)

    return similarity_score


def load(f, s):
    print("Loading fingerprints for authentication...")
    f_img = []
    s_img = []

    # load files from directories
    for file in os.listdir(f):
        if file.endswith('.png'):
            f_img.append(file)
        else:
            #print("file loading error")
            continue

    for file in os.listdir(s):
        if file.endswith('.png'):
            s_img.append(file)
        else:
            #print("file loading error")
            continue

    print((len(f_img) + len(s_img)), " files loaded")
    return f_img, s_img


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
    similarity_comparison = 163

    # compare each index of fingerprints to subject fingerprint to determine authentication:
    for i in range(total_fingerprints):
        # random number added to i, tests for true reject and false positive
        rand = [0, 1]
        rnum = random.choice(rand)
        if i < total_fingerprints-1:
            i2 = i + rnum
        else:
            i2 = i - rnum
        fingerprint_image = os.path.join(f_dirt, f_arr[i])
        subject_image = os.path.join(s_dirt, s_arr[i2])
        similarity_score = compare_fingerprints(fingerprint_image, subject_image)

        if similarity_score > similarity_comparison:
            # append tuple to authenticated list
            authenticated_fingerprints.append((fingerprint_image, subject_image))
            # test if true authentication / false authentication
            if fingerprint_image[-11:] == subject_image[-11:]:
                true_auth_counter += 1
            else:
                false_auth_counter += 1
        else:
            # append tuple to denied list
            denied_fingerprints.append((fingerprint_image, subject_image))
            if fingerprint_image[-11:] == subject_image[-11:]:
                false_reject_counter += 1
            else:
                true_deny_counter += 1

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


if __name__ == '__main__':
    # change the strings for f_dir and s_dir to the directory paths for those images.
    f_dir = "fingerprint"
    s_dir = 'subject'
    f_images, s_images = load(f_dir, s_dir)
    FRR, FAR, EER = compare_sets(f_dir, f_images, s_dir, s_images)
    print("--------------------")
    print("False Reject Rate:", FRR)
    print("False Accept Rate:", FAR)
    print("Equal Error Rate:", EER)


