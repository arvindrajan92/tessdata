""" 
Copyright (c) 2021-present Arvind Rajan

MIT License
"""
import os
import cv2
import argparse
import requests
import glob
import pickle
import time
import tqdm
import numpy as np
from shutil import copyfile
from PIL import Image


def preprocess(image, target=30):
    '''
    Preprocesses the image for Tesseract
    '''
    # correct the text height
    ratio = max(0.5, min(1.5, target / image.shape[0]))
    image = cv2.resize(image,
                       dsize=None,
                       fx=ratio,
                       fy=ratio,
                       interpolation=cv2.INTER_LANCZOS4)

    # apply Otsu thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert image if more than 3/5 of pixels are black
    height, width = thresh.shape
    if (cv2.countNonZero(thresh) / (height * width)) < (2 / 5):
        thresh = cv2.bitwise_not(thresh)

    return thresh

def configure_environment(n_cpu=None):
    '''
    Sets up the number of cores used by Tesseract for machines less than 4 cores.

            Parameters: 
                    n_cpu (int): A decimal integer for desired number of cpu to be used
    '''
    
    # check the input type
    if (n_cpu is not None) and (not isinstance(n_cpu, int)):
        raise TypeError

    # get the number of machine cpu
    cpu_count = os.cpu_count()

    # if n_cpu > cpu_count or cpu_count < 4, set OMP_THREAD_LIMIT to cpu_count
    if n_cpu is None:
        cpu_set = cpu_count
    else:
        cpu_set = cpu_count if (n_cpu > cpu_count) or (cpu_count < 4) else n_cpu

    # set OMP_THREAD_LIMIT variable
    print('Machine CPU count: {}\nSetting the number of threads to: {}'.format(cpu_count, cpu_set))
    os.environ['OMP_THREAD_LIMIT'] = str(cpu_set)


def download_models():
    '''
    Download different tessdata models and stores in models/ directory.
    '''
    # download the different tessdata models
    for model in ['tessdata', 'tessdata_best', 'tessdata_fast']:
        if not os.path.exists('models/' + model):
            os.makedirs('models/' + model)
        
        # eng model
        fname = 'models/' + model + '/eng.traineddata'
        if not os.path.isfile(fname):
            url = 'https://github.com/tesseract-ocr/' + model + '/raw/master/eng.traineddata'
            r = requests.get(url, allow_redirects=True)
            open(fname, 'wb').write(r.content)
            print('Downloaded {}'.format(fname))
        else:
            print('{} already exists'.format(fname))

        # osd model
        fname = 'models/' + model + '/osd.traineddata'
        if not os.path.isfile(fname):
            url = 'https://github.com/tesseract-ocr/' + model + '/raw/master/osd.traineddata'
            r = requests.get(url, allow_redirects=True)
            open(fname, 'wb').write(r.content)
            print('Downloaded {}'.format(fname))
        else:
            print('{} already exists'.format(fname))


def order_points(points):
    """
    Order the four corners of a box to follow the sequence of
    top left, top right, bottom right, and bottom left.
    """
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect.astype(np.int32)


def save_result(model, book, page, result):
    '''
    Saves the result as JSON file
    '''

    # result path
    result_path = 'results/' + model + '/' + book

    # create result folder if doesnt exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '/' + page + '.pickle', 'wb') as f:
        pickle.dump(result, f)
    
    
if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(description='Run performance analysis of the different Tesseract models.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--n_cpu', default=None, type=int, help='Number of CPUs for Tesseract engine to use (default: None)')
    args = parser.parse_args()

    # checking root privileges
    euid = os.geteuid()
    if euid != 0:
        import sys
        print("\nScript not started as root. Running sudo..")

        # replaces the currently-running process with the sudo
        args = ['sudo', sys.executable] + sys.argv + [os.environ]
        os.execlpe('sudo', *args)
    print('Running. Your euid is {}'.format(euid))

    # configure environment to be optimal for Tesseract engine
    # NOTE: to disable multithreading, set OMP_THREAD_LIMIT to 1
    print('\nSetting up environment...')
    configure_environment(n_cpu=args.n_cpu)

    # download Tesseract models
    print('\nDownloading tessdata models...')
    download_models()

    # get results
    print('\nRunning analysis...')
    for model in tqdm.tqdm(['tessdata', 'tessdata_fast', 'tessdata_best'], leave=False, position=0, desc='Models'):
        # replace the models
        copyfile(os.getcwd() + '/models/' + model + '/eng.traineddata', '/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata')
        copyfile(os.getcwd() + '/models/' + model + '/osd.traineddata', '/usr/share/tesseract-ocr/4.00/tessdata/osd.traineddata')

        # import Python's tesseract ocr library
        import tesserocr
        with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK, lang="eng", oem=tesserocr.OEM.LSTM_ONLY) as api:
            for book in tqdm.tqdm(os.listdir(args.data_dir), leave=False, position=1, desc='Books'):

                # get the images and bounding boxes list from images directory
                images_list = glob.glob(args.data_dir + book + '/gen_imgs/*.png')
                boxes_list = glob.glob(args.data_dir + book + '/gen_boxes/*.pickle')

                # show the image with bounding boxes drawn
                for image_path, box_path in tqdm.tqdm(zip(images_list, boxes_list), total=len(images_list), leave=False, position=2, desc='Pages'):

                    # read the image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    width, height = image.shape

                    # get text boxes from pickle file
                    with (open(box_path, "rb")) as picklefile:
                        text_boxes = pickle.load(picklefile)

                    # draw bounding boxes around the text and characters in different colours
                    result = []
                    for text_box in tqdm.tqdm(text_boxes, leave=False, position=3, desc='Texts'):

                        # ground truth
                        box_data = {k:v for k, v in text_box.items() if k in ['text', 'box']}

                        # fix the points
                        points = order_points(np.roll(box_data.get('box'), 1, axis=1))
                        box_data.update({'mod_box': (max(0, min(points[:, 0]) - 5),
                                                     max(0, min(points[:, 1]) - 5),
                                                     min(width, max(points[:, 0]) + 6),
                                                     min(height, max(points[:, 1]) + 6))})
                        
                        # sanity check on the values
                        mod_box = box_data.get('mod_box')
                        if (mod_box[0] > width) or (mod_box[1] > height) or (mod_box[2] > width) or (mod_box[3] > height):
                            continue
                        elif (mod_box[0] < 0) or (mod_box[1] < 0) or (mod_box[2] < 0) or (mod_box[3] < 0):
                            continue
                        elif (mod_box[0] == mod_box[2]) or (mod_box[1] == mod_box[3]):
                            continue

                        # crop the image
                        text_image = image[mod_box[1]:mod_box[3], mod_box[0]:mod_box[2]]

                        # preprocess
                        text_image = preprocess(text_image)

                        # get ocr output
                        try:
                            text_image = Image.fromarray(text_image)
                            start_time = time.time()
                            api.SetImage(text_image)
                            ocr_text = api.GetUTF8Text()
                            end_time = time.time()

                            # update dictionary
                            box_data.update({'ocr': {'text': ocr_text.replace('\n',''),
                                                    'time': end_time-start_time}})
                        except:
                            # update dictionary
                            box_data.update({'ocr': {'text': '',
                                                    'time': 0.0}})
          
                        result.append(box_data)

                    # save the results
                    save_result(model, book, (image_path.split('/')[-1]).split('.')[0], result)
        
        # lose access to tesserocr to reload again
        del tesserocr