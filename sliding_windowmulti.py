# CSEC471 Lab 4 - Group 10
# Comparing fingerprint images using the sliding window algoritnm

import os
import cv2
from matplotlib import pyplot
from skimage.morphology import skeletonize
from scipy.ndimage import center_of_mass
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy 
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def create_circular_mask(height, width, center=None, radius=None):
    if center is None:
        center = (int(width/2), int(height/2))
    if radius is None:
        radius = min(center[0], center[1], width-center[0], height-center[1])
    Y, X = numpy.ogrid[:height, :width]
    distance = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = distance <= radius
    return(mask)

def cleanup(path):
    # Clean up the image for comparison 
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(image)
    image = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11)
    return(image)

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = numpy.sum(arr[:, 0])
    sum_y = numpy.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def find_minutae(path, disp=False):
    #Find minutae in the finger print
    image = cleanup(path)
    timage = image//255
    image = skeletonize(timage, method='lee')
    com = center_of_mass(image)
    cmask = create_circular_mask(512, 512, com, 224)
    image[cmask==0] = 0
    if disp:
        pyplot.imshow(255-image, 'gray')
        pyplot.show()
    stepSize = 3
    (w_width, w_height) = (3,3)
    coords=[]
    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height]
            winmean = numpy.mean(window)
            if winmean in (8/9, 1/9):
                coords.append((x,y))
    coords = numpy.array(coords)
    coords_centr = centroidnp(coords)
    sort_coords = sorted(coords, key=lambda coord: numpy.linalg.norm(coord - coords_centr))
    return numpy.array(sort_coords[1:100])

def calculate_eer(real_score, fake_score):
    #Find equal error rate
    all_scores = numpy.concatenate([real_score, fake_score])
    labels = numpy.concatenate([numpy.ones_like(real_score), numpy.zeros_like(fake_score)])

    fpr, tpr, thresholds = metrics.roc_curve(labels, all_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)

    return eer, threshold

def calculate_far_frr(real_score, fake_score, threshold):
    #Find false acceptance rate and false error rate
    far = numpy.mean(fake_score>=threshold)
    frr = numpy.mean(real_score<threshold)
    return far, frr

def compare_prints(apath, bpath, threshold=7, debug=False):
    #Analyze the finger print
    minutae_a = find_minutae(apath, disp=debug)
    minutae_b = find_minutae(bpath, disp=debug)
    centroid_a = numpy.expand_dims(centroidnp(minutae_a), 0)
    centroid_b = numpy.expand_dims(centroidnp(minutae_b), 0)
    distance_a = numpy.linalg.norm(minutae_a - centroid_a[:, ], axis = 1)
    distance_b = numpy.linalg.norm(minutae_b - centroid_b[:, ], axis = 1)
    sort_distances = numpy.array(distance_a) - numpy.array(distance_b)
    
    #Calculate key values
    similarity = len(sort_distances[numpy.where(abs(sort_distances) < threshold)]) / len(sort_distances)
    eer, threshold = calculate_eer(distance_a, distance_b)
    far, frr = calculate_far_frr(distance_a, distance_b, threshold)
    
    return round(similarity, 2) , round(eer, 2), round(far, 2) , round(frr, 2)

def process_image_pair(file_a, file_b, fingerprint, subject):
    image_path_a = os.path.join(fingerprint, file_a)
    image_path_b = os.path.join(subject, file_b)
    similarity, eer, far, frr = compare_prints(image_path_a, image_path_b, threshold=7, debug=False)
    return similarity, eer, far, frr

def main(image_path_a, image_path_b):
    similarity_score = compare_prints(image_path_a, image_path_b, threshold=7, debug=False)

    print(f" {image_path_a} and {image_path_b}: {similarity_score}")


if __name__ == "__main__":
    # Replace these paths with the actual file paths of your images
    fingerprint = 'fingerprint'
    subject = 'subject'

    f_images = [f for f in os.listdir(fingerprint) if f.endswith('.png')]
    s_images = [f for f in os.listdir(subject) if f.endswith('.png')]

    similarity_scores = []
    eer_scores = []
    far_scores = []
    frr_scores = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        process_func = partial(process_image_pair, fingerprint=fingerprint, subject=subject)
        results = executor.map(process_func, f_images, s_images)
        for similarity, eer, far, frr in results:
                similarity_scores.append(similarity)
                eer_scores.append(eer)
                far_scores.append(far)
                frr_scores.append(frr)

    avg_similarity = round(numpy.mean(similarity_scores), 2)
    avg_eer = round(numpy.mean(eer_scores), 2)
    avg_far = round(numpy.mean(far_scores), 2)
    avg_frr = round(numpy.mean(frr_scores), 2)

    min_sim = round(min(similarity_scores), 2)
    min_eer = round(min(eer_scores), 2)
    min_far = round(min(far_scores), 2)
    min_frr = round(min(frr_scores), 2)

    max_sim = round(max(similarity_scores), 2)
    max_eer = round(max(eer_scores), 2)
    max_far = round(max(far_scores), 2)
    max_frr = round(max(frr_scores), 2)

    print("\nSimilarity:")
    print(f"Average Similarity Score: {avg_similarity}")
    print(f"Minimum Similarity Score: {min_sim}")
    print(f"Maximum Similarity Score: {max_sim}")
    
    print("\nEqual Error Rate:")
    print(f"Average Equal Error Rate: {avg_eer}")
    print(f"Minimum Equal Error Rate: {min_eer}")
    print(f"Maximum Equal Error Rate: {max_eer}")
    
    print("\nFalse Acceptance Rate:")
    print(f"Average False Acceptance Rate: {avg_far}")
    print(f"Minimum False Acceptance Rate: {min_far}")
    print(f"Maximum False Acceptance Rate: {max_far}")
    
    print("\nFalse Rejection Rate:")
    print(f"Average False Rejection Rate: {avg_frr}")
    print(f"Minimum False Rejection Rate: {min_frr}")
    print(f"Maximum False Rejection Rate: {max_frr}")

