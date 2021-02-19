""" 
Copyright (c) 2021-present Arvind Rajan

MIT License
"""
import os
import argparse
import pickle
import numpy as np
import Levenshtein as lev

from matplotlib import pyplot as plt
from statistics import median, mean, stdev

def extract_results(result_files, levenshtein=False):
    '''
    Extracts results from the pickle file
    '''
    match, time = [], []
    for result_file in result_files:
        # read the pickle file
        with open(result_file, 'rb') as f:
            result = pickle.load(f)
        
        # loop over all the texts
        for text in result:
            ocr = text.get('ocr')
            # get match and time if ocr result exists
            if ocr is not None:
                # depends if levenshtein distance or exact match requested
                if levenshtein is True:
                    match.append(lev.ratio(ocr.get('text').lower(), text.get('text').lower()))
                else:
                    match.append(ocr.get('text').lower() == text.get('text').lower())
                
                # processing time - doesn't matter if levenshtein distance is requested
                time.append(ocr.get('time'))

    return match, time


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='Run performance analysis of the different Tesseract models.')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory of the dataset')
    args = parser.parse_args()

    # get the location of all pickle files
    all_results = []
    for root, dirs, files in os.walk(os.path.join(args.result_dir)):
        for name in files:
            if '.pickle' in name:
                all_results.append(os.path.join(root, name))

    # tessdata
    print('\nExtracting tessdata results...')
    results = [result for result in all_results if '/tessdata/' in result]
    tessdata_ocr_matches, tessdata_ocr_times = extract_results(results)

    # tessdata_best
    print('Extracting tessdata_best results...')
    results = [result for result in all_results if '/tessdata_best/' in result]
    tessdata_best_ocr_matches, tessdata_best_ocr_times = extract_results(results)

    # tessdata_fast
    print('Extracting tessdata_fast results...')
    results = [result for result in all_results if '/tessdata_fast/' in result]
    tessdata_fast_ocr_matches, tessdata_best_ocr_times = extract_results(results)

    # plot accuracy results
    accuracy_result = [
        sum(tessdata_ocr_matches)/len(tessdata_ocr_matches),
        sum(tessdata_best_ocr_matches)/len(tessdata_best_ocr_matches),
        sum(tessdata_fast_ocr_matches)/len(tessdata_fast_ocr_matches)
    ]

    # tessdata
    print('\nExtracting tessdata results based on levenshtein distance...')
    results = [result for result in all_results if '/tessdata/' in result]
    tessdata_ocr_matches, tessdata_ocr_times = extract_results(results, levenshtein=True)

    # tessdata_best
    print('Extracting tessdata_best results based on levenshtein distance...')
    results = [result for result in all_results if '/tessdata_best/' in result]
    tessdata_best_ocr_matches, tessdata_best_ocr_times = extract_results(results, levenshtein=True)

    # tessdata_fast
    print('Extracting tessdata_fast results based on levenshtein distance...')
    results = [result for result in all_results if '/tessdata_fast/' in result]
    tessdata_fast_ocr_matches, tessdata_fast_ocr_times = extract_results(results, levenshtein=True)

    # remove 1s and outliers
    tessdata_ocr_matches = [r for r in tessdata_ocr_matches if r != 1.0]
    data_mean, data_std = mean(tessdata_ocr_matches), stdev(tessdata_ocr_matches)
    tessdata_ocr_matches_clean = [r for r in tessdata_ocr_matches if abs(r - data_mean) < (2 * data_std)]

    tessdata_best_ocr_matches = [r for r in tessdata_best_ocr_matches if r != 1.0]
    data_mean, data_std = mean(tessdata_best_ocr_matches), stdev(tessdata_best_ocr_matches)
    tessdata_best_ocr_matches_clean = [r for r in tessdata_best_ocr_matches if abs(r - data_mean) < (2 * data_std)]

    tessdata_fast_ocr_matches = [r for r in tessdata_fast_ocr_matches if r != 1.0]
    data_mean, data_std = mean(tessdata_fast_ocr_matches), stdev(tessdata_fast_ocr_matches)
    tessdata_fast_ocr_matches_clean = [r for r in tessdata_fast_ocr_matches if abs(r - data_mean) < (2 * data_std)]

    # plot the figures
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Levenshtein distance')
    ax.boxplot([tessdata_ocr_matches_clean, tessdata_best_ocr_matches_clean, tessdata_fast_ocr_matches_clean])
    plt.xticks([1, 2, 3], ['tessdata', 'tessdata_best', 'tessdata_fast'])
    plt.grid(True)
    plt.show()
    plt.xlabel('Language data')
    plt.ylabel('Score')
    plt.savefig('boxplot.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('OCR time')
    ax.boxplot([tessdata_ocr_times, tessdata_best_ocr_times, tessdata_fast_ocr_times])
    plt.xticks([1, 2, 3], ['tessdata', 'tessdata_best', 'tessdata_fast'])
    plt.grid(True)
    plt.show()
    plt.xlabel('Language data')
    plt.ylabel('Time (s)')
    plt.savefig('time.png')

    fig = plt.figure(figsize = (10, 5))
    plt.bar(['tessdata', 'tessdata_best', 'tessdata_fast'], accuracy_result) 
    plt.title("OCR accuracy") 
    plt.ylim(0.85, 0.95)
    plt.grid(True)
    plt.show() 
    plt.xlabel('Language data')
    plt.ylabel('Score')
    plt.savefig('accuracy.png')
