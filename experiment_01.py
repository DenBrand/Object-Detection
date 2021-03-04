import argparse
from cv2 import CascadeClassifier
from experimentutils import LoadYolo, RunDetection, LoadCascadeGroundTruth, \
    LoadYoloGroundTruth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program performs the '
                                     'first experiment of Dennis Brands\' '
                                     'bachelor\'s thesis: fixed vs arbitrary '
                                     'Object Colors')
    parser.add_argument('--test-path',
                        required=True,
                        help='path to the directory containing test images',
                        dest='testImgsDir')
    parser.add_argument('--test-names',
                        required=True,
                        help='path to a TXT file listing all test img names'
                        'in test-path directory',
                        dest='testImgsNames')
    parser.add_argument('--cascades',
                        required=True,
                        nargs='+',
                        help='paths to the XML files of the cascade '
                        'classifiers',
                        dest='cascadeXMLs')
    parser.add_argument('--truth-cascade',
                        required=True,
                        help='path to the directory containing ground-truth '
                        'JSON files for cascade classifiers',
                        dest='cascadeTruth')
    parser.add_argument('--min-size',
                        default=30,
                        type=int,
                        help='minimal object size along x- and y-axis',
                        dest='minSize')
    parser.add_argument('--yolo',
                        required=True,
                        help='path to the weights file of the yolo model',
                        dest='yoloWeights')
    parser.add_argument('--cfg',
                        required=True,
                        help='path to the config file of the yolo model',
                        dest='yoloConfig')
    parser.add_argument('--yolo-truth',
                        required=True,
                        help='path to the directory containing ground-truth '
                        'yolo labels',
                        dest='yoloTruth')
    args = parser.parse_args()
    
    # load models
    cascade_detectors = [CascadeClassifier(XML) for XML in args.cascadeXMLs]
    yolo_net, output_layers = LoadYolo(args.yoloWeights, args.yoloConfig)
    
    # perform actual detection and get detections
    cascade_rects, yolo_rects = RunDetection(args.testImgsDir,
                 args.testImgsNames,
                 cascade_detectors,
                 args.minSize,
                 yolo_net,
                 output_layers)
    
    # DEBUG
    print(yolo_rects)
    
    # read in ground-truth data for cascade classifier
    cubes_pos, balls_pos = LoadCascadeGroundTruth(args.cascadeTruth)
    
    # read in ground-truth data for yolov5
    yolo_pos = LoadYoloGroundTruth(args.yoloTruth)
    
    
    
    
    
    
    
    