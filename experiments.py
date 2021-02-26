import cv2
import os
from os.path import join
from os import listdir
import json
import shutil
from random import randrange
from typing import Tuple, List

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

def SplitData(img_dir: str,
              cascade_dir: str,
              yolo_dir: str,
              dest_img_dir: str,
              dest_cascade_dir: str,
              dest_yolo_dir: str,
              test_set_size: int,
              chosen_imgs: str=None):
    """Split the data set given by image directory, with cascade classifier
    labels in json_dir and yolo labels in yolo_dir and put test_set_size
    samples for the test set.
    
    Keyword arguments:
    img_dir -- the path to the directory the images are in
    cascade_dir -- the path to the directory, containing jsons labeling data
    for the cascade classifier
    yolo_dir -- the path to the directory containing the yolo labels
    test_set_dir -- the path to the directory where all test set data should be
    stored
    test_set_size -- number of samples for the test data set
    chosen_imgs -- if specified, images in this txt will be chosen, instead of
    randomly picking test_set_size elemets out of img_dir directory
    """
    
    # get file names, filter wrong out
    img_list = [img for img in listdir(img_dir) if img[-4:] == '.png']
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
    # DEBUG:
    test_set_imgs.sort()
    
    # get run_data copy them into thest_set_data and delete chosen images
    os.makedirs(dest_cascade_dir, exist_ok=True)
    for json_name in json_list:
        shutil.copy(join(cascade_dir, json_name), dest_cascade_dir)
        
        def RemoveEntries(orig_or_new: bool):
            """orig_or_new -- True -> handle source data
            False -> handle destination data
            """
            
            # specity json_path
            json_path = None
            if orig_or_new:
                json_path = cascade_dir
            else:
                json_path = dest_cascade_dir
                
            data = None
            with open(join(json_path, json_name), 'r') as json_file:
                data = json.load(json_file)
                
            def RemoveEntriesHelper(detec_cls: str, label_type: str):
                for run_data in data['runData']:
                    for entry in run_data[detec_cls][label_type]:
                        if orig_or_new:
                            if entry['path'].replace(label_type + '\\', '') in test_set_imgs:
                                run_data[detec_cls][label_type].remove(entry)
                        else:
                            if entry['path'].replace(label_type + '\\', '') not in test_set_imgs:
                                run_data[detec_cls][label_type].remove(entry)
                                
            RemoveEntriesHelper('cubes', 'positives')
            RemoveEntriesHelper('cubes', 'negatives')
            RemoveEntriesHelper('balls', 'positives')
            RemoveEntriesHelper('balls', 'negatives')
                                
            # write back
            with open(join(json_path, json_name), 'w') as json_file:
                json.dump(data, json_file, indent=4)
                
        RemoveEntries(True)
        RemoveEntries(False)
        
        # with open(join(cascade_dir, json_name), 'r+') as json_file:
        #     data = json.load(json_file)
        #     def RemoveEntriesHelper(detec_cls: str, label_type: str):
        #         for run_data in data['runData']:
        #             for entry in run_data[detec_cls][label_type]:
        #                 if entry['path'].replace(label_type + '\\', '') \
        #                 in test_set_imgs:
        #                     run_data[detec_cls][label_type].remove(entry)
        #     RemoveEntriesHelper('cubes', 'positives')
        #     RemoveEntriesHelper('cubes', 'negatives')
        #     RemoveEntriesHelper('balls', 'positives')
        #     RemoveEntriesHelper('balls', 'negatives')

        #     # write back
        #     json.dump(data, json_file, indent=4)
    
        # with open(join(dest_cascade_dir, json_name), 'r+') as json_file:
        #     data = json.load(json_file)
        #     def RemoveEntriesHelper(detec_cls: str, label_type: str):
        #         for run_data in data['runData']:
        #             for entry in run_data[detec_cls][label_type]:
        #                 if entry['path'].replace(label_type + '\\', '') \
        #                 not in test_set_images:
        #                     run_data[detec_cls][label_type].remove(entry)
        #     RemoveEntriesHelper('cubes', 'positives')
        #     RemoveEntriesHelper('cubes', 'negatives')
        #     RemoveEntriesHelper('balls', 'positives')
        #     RemoveEntriesHelper('balls', 'negatives')
            
        #     json.dump(data, json_file, indent=4)
            
    # move files
    for img in test_set_imgs:
        # move image -> test_set_dir
        os.makedirs(dest_img_dir, exist_ok=True)
        src = join(img_dir, img)
        dest = join(dest_img_dir, img)
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
    """Same as SplitData, but with test_set_portion (in percent instead of
    an absolute number)"""
    
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
    SplitData(r'experiment_01-arbitrary_colors\training_data\model_fixed\raw_data',
              r'experiment_01-arbitrary_colors\training_data\cascade_JSONs',
              r'experiment_01-arbitrary_colors\training_data\yolov5_labels',
              r'experiment_01-arbitrary_colors\test_data\images',
              r'experiment_01-arbitrary_colors\test_data\cascade_labels',
              r'experiment_01-arbitrary_colors\test_data\yolo_labels',
              400)