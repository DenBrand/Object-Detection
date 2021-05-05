import cv2
import os
import json
import shutil
from os.path import join
from os import listdir
from random import randrange
from itertools import product

def LoadYolo(weights, cfg):
    """Load yolo detector and return it and it's output layers.
    
    Keyword arguments:
    weights -- the path to the weights file
    cfg -- the path to the config file
    """

    detector = cv2.dnn.readNet(weights, cfg)

    layers_names = detector.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in detector.getUnconnectedOutLayers()]

    return detector, output_layers

def DetectYolo(img, detector, output_layers, bx, by):
    """REWRITE --- openCV can't handle .pt file --- Perform yolo detection on img and return the outputs
    
    Keyword arguments:
    img -- the image object (opencv) to perform detection on
    detector -- the loaded yolo detector
    output_layers -- the output_layers of the yolo detector
    bx, by -- the size of the blob which should be fed into the detector
    """
    
    blob = cv2.dnn.blobFromImage(   img,
                                    scalefactor=1/255,
                                    size=(bx, by),
                                    mean=(0, 0, 0),
                                    swapRB=True,
                                    crop=False)
    detector.setInput(blob)
    outputs = detector.forward(output_layers)

    return outputs

def DetectCascade(cascade, imgs, minimalObjectSize):
    """DEPRECATED --- Run detection with cascade classifier on images and return bounding
    boxes and confidences.
    
    Keyword arguments:
    cascade -- cascade model (already loaded from XML)
    imgs -- list of paths to test images
    minimalObjectSize -- minimal width or height of present objects
    
    
    all_preds = []
    for img in imgs:
        img_loaded = cv2.imread(img)
        
        # one file for each image, listing preds like:
        # <class_id> <conf> <left> <top> <width> <height> (all absolute)
        objects, rejectLevels, levelWeights = cascade.detectMultiScale3(img_loaded,
                                                                        scaleFactor=1.1,
                                                                        minNeighbors=5,
                                                                        minSize=(30, 30),
                                                                        flags = cv2.CASCADE_SCALE_IMAGE,
                                                                        outputRejectLevels = True)
        
        img_name = img[img.rfind('/')+1:]
        """

def RunDetection(image_set_path,
            image_names,
            cascade_classifiers,
            minimal_object_size,
            yolov5_net,
            yolov5_output_layers):
    """Run detection and return resulting boundary boxes."""
    
    # cascade classifier
    cascade_rects = {}
    yolo_rects = {}
    for img_name in image_names:
        img = cv2.imread(join(image_set_path, img_name))
        
        # cascade classifier
        rects = []
        for classifier in cascade_classifiers:
            rects.append(classifier.detectMultiScale(img, minSize=minimal_object_size))
        cascade_rects[img_name] = rects
        
        # yolov5
        rects = DetectYolo(img, yolov5_net, yolov5_output_layers)
        yolo_rects[img_name] = rects
        
        
    return cascade_rects, yolo_rects

def LoadCascadeGroundTruth(cascade_truth):
    """DEPRECATED"""
    # get file names
    json_paths = listdir(cascade_truth)
    
    # get all positives
    cubes_positives = []
    balls_positives = []
    for path in json_paths:
        data = None
        with open(join(cascade_truth, path)) as file:
            data['runData'] = json.load(file)
            
        for run_data in data:
            cubes_positives.extend(run_data['cubes']['positives'])
            balls_positives.extend(run_data['balls']['positives'])
            
    return cubes_positives, balls_positives

def LoadYoloGroundTruth(yolo_truth):
    """DEPRECATED"""
    # get file names
    label_names = listdir(yolo_truth)
    
    # get all positives
    positives = {}
    for name in label_names:
        positives[name] = []
        with open(join(yolo_truth, name), 'r') as file:
            for line in file.readlines():
                positives[name].append(line)
            
        # sometimes there is an empty line at the end of the file
        if len(positives[name][-1]) == 0:
            positives[name].pop(-1)
            
    return positives
    
# def CountTpFpTnNn(classification_result: DetectionResults):
    # """Take results and count true-positives, false-positives, true-negatives
    # and false-negatives."""
    
    # pass

def SplitData(img_dir_1: str,
              img_dir_2: str,
              cascade_dir: str,
              yolo_dir: str,
              dest_img_dir_1: str,
              dest_img_dir_2: str,
              suffix_of_set_2: str,
              dest_cascade_dir: str,
              dest_yolo_dir: str,
              test_set_size: int,
              chosen_imgs: str=None):
    """Split the data sets given by img_dir_1 and img_dir_2, with cascade
    classifier labels in cascade_dir and yolo labels in yolo_dir and chose
    test_set_size samples for the test set. chosen_imgs specifies a file where
    chosen images could be read in of instead. Images from img_dir_1 and 2 are
    moved respectively into dest_img_dir_1 and 2. suffix_of_set_2 specifies,
    what string must be added to images names in img_dir_1, to refer to their
    counterpart in img_dir_2.
    
    Keyword arguments:
    img_dir_1 -- the path to the directory the images from set 1 are in
    img_dir_2 -- the path to the directory the images from set 2 are in
    cascade_dir -- the path to the directory, containing jsons labeling data
    for the cascade classifier
    yolo_dir -- the path to the directory containing the yolo labels
    test_set_dir -- the path to the directory where all test set data should be
    stored
    test_set_size -- number of samples for the test data set
    chosen_imgs -- if specified, images in this txt will be chosen, instead of
    randomly picking test_set_size elements out of img_dir_1 directory
    """
    
    # get file names, filter wrong out
    img_list_1 = [img for img in listdir(img_dir_1) if img[-4:] == '.png']
    img_list_2 = [img for img in listdir(img_dir_2) if img[-4:] == '.png']
    json_list = [json for json in listdir(cascade_dir) if json[-5:] == '.json']
    
    assert len(img_list_1) == len(img_list_2), ''
    assert test_set_size < len(img_list_1), 'you need training data to '
    'train a model'
    
    test_set_imgs = []
    if chosen_imgs is None:
        # choose test data samples randomly
        for _ in range(test_set_size):
            idx = randrange(0, len(img_list_1))
            test_set_imgs.append(img_list_1.pop(idx))
    else:
        # load test data samples according to chosen_imgs
        with open(chosen_imgs, 'r') as chosen_file:
            test_set_imgs = [line.replace('\n', '') for line in chosen_file]
    
    # get run_data copy them into thest_set_data and delete chosen images
    os.makedirs(dest_cascade_dir, exist_ok=True)
        
    def RemoveEntries(orig_or_new: bool):
        """orig_or_new -- True -> handle source data
        False -> handle destination data
        """
        
        for json_name in json_list:
            if orig_or_new:
                shutil.copy(join(cascade_dir, json_name), dest_cascade_dir)
            
            # specity json_path
            json_path = None
            if orig_or_new:
                json_path = cascade_dir
            else:
                json_path = dest_cascade_dir
                
            data = None
            with open(join(json_path, json_name), 'r') as json_file:
                data = json.load(json_file)
                
            def RemoveEntriesHelper():
                for run_data in data['runData']:
                    for detec_cls, label_type in product(('cubes', 'balls'),
                                         ('positives', 'negatives')):
                        current_block = run_data[detec_cls][label_type]
                        for entry in reversed(current_block):
                            if orig_or_new:
                                if entry['path'].replace(label_type+'\\', '') \
                                    in test_set_imgs:
                                    current_block.remove(entry)
                            else:
                                if entry['path'].replace(label_type+'\\', '') \
                                    not in test_set_imgs:
                                    current_block.remove(entry)
            RemoveEntriesHelper()
            
            # write back
            with open(join(json_path, json_name), 'w') as json_file:
                json.dump(data, json_file, indent=4)
                
    RemoveEntries(True)
    RemoveEntries(False)
            
    # move files
    for img in test_set_imgs:
        # move image -> dest_img_dir_1
        os.makedirs(dest_img_dir_1, exist_ok=True)
        src = join(img_dir_1, img)
        dest = join(dest_img_dir_1, img)
        os.rename(src, dest)
        
        # move counterpart image -> dest_img_dir_2
        os.makedirs(dest_img_dir_2, exist_ok=True)
        src = join(img_dir_2, img.replace('.png', suffix_of_set_2 + '.png'))
        dest = join(dest_img_dir_2, img.replace('.png',
                                                suffix_of_set_2 + '.png'))
        os.rename(src, dest)
        
        # move yolo label -> test_set_dir
        os.makedirs(dest_yolo_dir, exist_ok=True)
        label_name = img.replace('.png', '.txt')
        src = join(yolo_dir, label_name)
        dest = join(dest_yolo_dir, label_name)
        os.rename(src, dest)
            
def SplitDataByPercentage(img_dir_1: str,
              img_dir_2: str,
              cascade_dir: str,
              yolo_dir: str,
              dest_img_dir_1: str,
              dest_img_dir_2: str,
              suffix_of_set_2: str,
              dest_cascade_dir: str,
              dest_yolo_dir: str,
              test_set_portion: int,
              chosen_imgs: str=None):
    """Same as SplitData, but with test_set_portion (in percent instead of an
    absolute number)"""
    
    img_list_1 = [img for img in listdir(img_dir_1) if img[-4:] == '.png']
    
    size = len(img_list_1) * test_set_portion // 100
            
    SplitData(img_dir_1,
              img_dir_2,
              cascade_dir,
              yolo_dir,
              dest_img_dir_1,
              dest_img_dir_2,
              suffix_of_set_2,
              dest_cascade_dir,
              dest_yolo_dir,
              size,
              chosen_imgs)
    