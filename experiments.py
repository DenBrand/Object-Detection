import cv2
from typing import Tuple, List

class DetectionResults:
    """In this class the cascade classifier and yolov5 finally become things
    of the same kind. At least I hope so...
    """
    
    def __init__(self, ):
        # TODO
        pass
        
def RunTest(images_sets_paths: Tuple[str],
            images_names: Tuple[str],
            ground_truth_cascade: List[str],
            ground_truth_yolov5: List[str],
            models_cascade: tuple[CascadeClassifier],
            model_yolov5: dnn_Net):
    """Run experiment and return its results."""
    
    # cascade classifier
    # TODO
    
    # yolov5
    
    return {
        'cascade': {
            
        },
        'yolov5': {
                
        }
    }
    
def CountTpFpTnNn(classification_result: DetectionResults):
    """Take results and count true-positives, false-positives, true-negatives
    and false-negatives."""
    
    pass

def 

if __name__ == '__main__':
    pass