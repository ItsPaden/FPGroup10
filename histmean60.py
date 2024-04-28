# Ethan Zeevi - CSEC471 Lab 4 - Group 10
# Comparing fingerprint images using the histmean60 algoritnm and ImageChops

import os
import random
from PIL import Image, ImageChops
import math

# Compares 2 fingerprint image files
# imageA, imageB: filepath to fingerprint images
# Returns similarity expressed as a percentage
def compare_prints(imageA, imageB):
    imA = Image.open(imageA)
    imB = Image.open(imageB)
    h = ImageChops.difference(imA, imB).histogram()
    val = math.sqrt(sum(h*(i**2) for i, h in enumerate(h)) / (float(imA.size[0]) * imB.size[1]))
    difference = val/100
    return(1-difference)

# Loads image files from directory to associated f/s lists
# dir: directory (relative or absolute path works) for image files
# Returns 2 lists for the f and s images
def load(dir):
    print("Loading fingerprints for authentication...")
    f_img = []
    s_img = []

    for file in os.listdir(dir):
        if file.startswith('f') and file.endswith('.png'):
            f_img.append(file)

        elif file.startswith('s') and file.endswith('.png'):
            s_img.append(file)

    print((len(f_img) + len(s_img)), " files loaded")
    return f_img, s_img

# Compares either a matching f/s pair or a non-matching f/s pair and records false/true positives and false/true negatives
# dir: directory (relative or absolute path works) for image files, f_arr: list of f image files, s_arr: list of s image files
# Returns false reject rate (FRR), false accept rate (FAR), and equal error rate (EER)
def compare_sets(dir, f_arr, s_arr):
    # Compare fingerprints vars
    total_fingerprints = len(f_arr)
    authenticated_fingerprints = []
    denied_fingerprints = []
    true_auth_counter = 0
    true_deny_counter = 0
    false_auth_counter = 0
    false_reject_counter = 0

    # determines the similarity level needed to accept the comparison from 0-1
    # high value = high strictness
    # 0.50 for low tolerance
    # 0.40 for high tolerance without excessive error rate
    accept_threshold = 0.40

    # compare each index of fingerprints to subject fingerprint to determine authentication:
    for i in range(total_fingerprints):
        # random number added to i, tests for true reject and false positive
        rand = [0, 1]
        rnum = random.choice(rand)
        if i < total_fingerprints-1:
            i2 = i + rnum
        else:
            i2 = i - rnum
        fingerprint_image = os.path.join(dir, f_arr[i])
        subject_image = os.path.join(dir, s_arr[i2])
        similarity_score = compare_prints(fingerprint_image, subject_image)

        # If similarity > acceptance threshold -> accept as authenticated
        if similarity_score > accept_threshold:
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
    # Change dir to the directory containing the 4000 fingerprint images
    dir = "FingerprintData"
    f_images, s_images = load(dir)
    FRR, FAR, EER = compare_sets(dir, f_images, s_images)
    print("--------------------")
    print("False Reject Rate:", FRR)
    print("False Accept Rate:", FAR)
    print("Equal Error Rate:", EER)
