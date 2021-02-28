#import cv2
import os
from os.path import join
from os import listdir
import json
import shutil
from copy import deepcopy
from random import randrange
from typing import Tuple, List
from itertools import product

# class DetectionResults:
#     """In this class the cascade classifier and yolov5 finally become things
#     of the same kind. At least I hope so...
#     """
    
#     def __init__(self, ):
#         # TODO
#         pass
        
# def RunTest(images_sets_paths: Tuple[str],
#             images_names: Tuple[str],
#             ground_truth_cascade: List[str],
#             ground_truth_yolov5: List[str],
#             models_cascade: tuple[CascadeClassifier],
#             model_yolov5: dnn_Net):
#     """Run experiment and return its results."""
    
#     # cascade classifier
#     # TODO
    
#     # yolov5
    
#     return {
#         'cascade': {
#             # TODO
#         },
#         'yolov5': {
#             # TODO 
#         }
#     }
    
# def CountTpFpTnNn(classification_result: DetectionResults):
#     """Take results and count true-positives, false-positives, true-negatives
#     and false-negatives."""
    
#     pass

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
    img_dir_2 -- teh path to the directory the images from set 2 are in
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
    img_list = [img for img in listdir(img_dir_1) if img[-4:] == '.png']
    json_list = [json for json in listdir(cascade_dir) if json[-5:] == '.json']
    
    assert test_set_size < len(img_list), 'you need training to data to '
    'train a model'
    
    test_set_imgs = []
    if chosen_imgs is None:
        # choose test data samples randomly
        for _ in range(test_set_size):
            idx = randrange(0, len(img_list))
            test_set_imgs.append(img_list.pop(idx))
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
            
def SplitDataByPercentage(img_dir: str,
                          cascade_dir: str,
                          yolo_dir: str,
                          dest_img_dir: str,
                          dest_cascade_dir: str,
                          dest_yolo_dir: str,
                          test_set_portion: int,
                          chosen_imgs: str=None):
    """Same as SplitData, but with test_set_portion (in percent instead of an
    absolute number)"""
    
    img_list = [img for img in listdir(img_dir) if img[-4:] == '.png']
    size = len(img_list) * test_set_portion // 100
            
    SplitData(img_dir,
              cascade_dir,
              yolo_dir,
              dest_img_dir,
              dest_cascade_dir,
              dest_yolo_dir,
              size,
              chosen_imgs)

if __name__ == '__main__':    
    # Split for experiment 1:
    # SplitData(join('experiment_01-arbitrary_colors', 'training_data',
    #                'model_fixed', 'raw_data'),
    #           join('experiment_01-arbitrary_colors', 'training_data',
    #                'model_arbitrary', 'raw_data'),
    #           join('experiment_01-arbitrary_colors', 'training_data',
    #                'cascade_JSONs'),
    #           join('experiment_01-arbitrary_colors', 'training_data',
    #                'yolov5_labels'),
    #           join('experiment_01-arbitrary_colors', 'test_data',
    #                'images_fixed'),
    #           join('experiment_01-arbitrary_colors', 'test_data',
    #                'images_arbitrary'),
    #           '_randomized_colors',
    #           join('experiment_01-arbitrary_colors', 'test_data',
    #                'cascade_labels'),
    #           join('experiment_01-arbitrary_colors', 'test_data',
    #                'yolo_labels'),
    #           400)