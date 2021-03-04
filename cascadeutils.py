import os
import os.path
import cv2
import math
import json
import shutil
import platform

# new classes need to be added here
DETECTABLE_CLASSES = ['cubes', 'balls', 'tetraeders']

def CreateNegTxt(detectable_class_id: int):
    """DEPRECATED: FetchNewTrainingData does that for us now."""
    
    negs_dir_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'negatives')
    negs_txt_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'negs.txt')

    negs_paths = os.listdir(negs_dir_path)
    negs_paths = [os.path.join(negs_dir_path, file_name) for file_name in negs_paths]

    with open(negs_txt_path, 'a') as negTxt:
        for path in negs_paths:

            path = path.replace(DETECTABLE_CLASSES[detectable_class_id] + '_data' + '\\', '') \
                        .replace(DETECTABLE_CLASSES[detectable_class_id] + '_data' + '/', '')

            negTxt.write(path + '\n')

def CreatePosTxtAndFilterNegsOut(detectable_class_id: int):
    """DEPRECATED: FetchNewTrainingData does that for us now."""
    
    poss_dir_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'positives')
    labels_dir_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'labels')
    negs_dir_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'negatives')
    poss_txt_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'pos.txt')

    labels_paths = os.listdir(labels_dir_path)
    labels_paths = [os.path.join(labels_dir_path, path) for path in labels_paths]

    # iterate over all label files
    all_label_data = []
    for labels_path in labels_paths:
        with open(labels_path, 'r') as labels:
            image_path = labels_path.replace(labels_dir_path, poss_dir_path).replace('.txt', '.jpg')
            image = cv2.imread(image_path)
            im_width, im_height, _ = image.shape
            line_head = [image_path]
            lines = labels.readlines()

            # iterate over all labels
            line_tail = []
            for line in lines:
                try:
                    id, x, y, w, h = line.split()
                    id = int(id)
                    w = float(w) * im_width
                    h = float(h) * im_height
                    x = int(math.ceil(float(x) * im_width - w / 2)) - 4
                    y = int(math.ceil(float(y) * im_height - w / 2)) - 4
                    w = int(math.ceil(w)) + 8
                    h = int(math.ceil(h)) + 8

                    # check if label makes sense
                    legit = True
                    if math.ceil(x + w) > im_width \
                        or  math.ceil(y + h) > im_height \
                        or x < 3 or x + w > im_width - 3 \
                        or y < 3 or y + h > im_height - 3 \
                        or w < 6 or h < 6:
                        legit = False

                    if id == detectable_class_id and legit:
                        line_tail.append(' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))

                except Exception as err:
                    print(err)
                    print(image_path + ' will be transfered to ' + negs_dir_path)
                    src_path = image_path
                    dest_path = image_path.replace('positives', 'negatives')
                    os.replace(src_path, dest_path)
                    line_tail.append('')
            line_head.append(line_tail)
        all_label_data.append(line_head)
    
    with open(poss_txt_path, 'a') as poss_txt:
        for label_file_data in all_label_data:
            if len(label_file_data[1]) > 0:
                line = label_file_data[0].replace('custom_data\\', '').replace('custom_data/', '') + ' ' + str(len(label_file_data[1]))
                for data in label_file_data[1]:
                    line += data
                poss_txt.write(line + '\n')
            else:
                print(label_file_data[0] + ' will be transfered to ' + negs_dir_path)
                src_path = label_file_data[0]
                dest_path = label_file_data[0].replace('positives', 'negatives')
                os.replace(src_path, dest_path)

def FetchNewTrainingData(src_dir: str, imgs_dir: str, target_dir: str, img_name_ending: str=None):
    """Read in run data from all JSONs in src_dir, generate pos.txt and
    neg.txt in target_dir and sort (copy) samples in imgs_dir into
    target_dir/positives/ and target_dir/negatives.
    
    Keyword arguments:
    src_dir -- the path to the directory in which the JSONs containing
    training data are
    imgs_dir -- the path to the directory in which the images (positives
    and negatives) are
    target_dir -- the path to the directory in which pos.txt, neg.txt,
    positives/ and negatives/ shall be generated in (sorted by detectable
    classes). already_fetched.txt will be generated there too, to prevent
    multiple fetching.
    img_name_ending -- normally the image names end with <id>.png,
    but for example in the arbitrary colors data set, they will end with
    <id>_randomized_colors.png. In this case, you should pass
    '_randomized_colors'
    """
    
    # helping function
    def Fetch(run_data):
        pos_txt_name = 'pos.txt'
        neg_txt_name = 'neg.txt'

        for detectable_class in DETECTABLE_CLASSES:
            if detectable_class in run_data.keys():

                # get positives and negatives of this detectable class
                positives = run_data[detectable_class]['positives']
                negatives = run_data[detectable_class]['negatives']

                # fetch positives
                positives_dict = {}
                for pos_entry in positives:
                    if not pos_entry['path'] in positives_dict.keys():
                        positives_dict[pos_entry['path']] = [pos_entry['boxEntry']]

                    else:
                        positives_dict[pos_entry['path']].append(pos_entry['boxEntry'])

                # ensure directory for this detectable class exists
                os.makedirs(os.path.join(   target_dir, detectable_class + '_data', 'positives'),
                                            exist_ok=True)
                os.makedirs(os.path.join(   target_dir, detectable_class + '_data', 'negatives'),
                                            exist_ok=True)

                # finally write positive entries in pos.txt
                with open(os.path.join(target_dir, detectable_class + '_data', pos_txt_name), 'a+') as pos_txt:
                    for path, entries in positives_dict.items():
                        num_of_entries = len(entries)
                        
                        if img_name_ending is not None:
                            path = path.replace('.png', img_name_ending + '.png')

                        line = path + ' ' + str(num_of_entries)
                        for entry in entries:
                            for key in entry.keys():
                                line += ' ' + str(entry[key])

                        # write
                        pos_txt.write(line + '\n')

                        # copy file into positives directory
                        file_name = path.replace('positives\\', '')
                        src_img_path = os.path.join(imgs_dir, file_name)
                        dst_img_path = os.path.join(target_dir,
                                                    detectable_class + '_data',
                                                    'positives',
                                                    file_name)
                        shutil.copyfile(src_img_path, dst_img_path)

                # write negative entries into neg.txt, move corresponding images
                with open(os.path.join(target_dir, detectable_class + '_data', neg_txt_name), 'a+') as neg_txt:
                    for path_dict in negatives:

                        path = path_dict['path']
                        if img_name_ending is not None:
                            path = path.replace('.png', img_name_ending + '.png')

                        # write
                        neg_txt.write(path + '\n')

                        # copy file into negatives directory
                        file_name = path.replace('negatives\\', '')
                        src_img_path = os.path.join(imgs_dir, file_name)
                        dst_img_path = os.path.join(target_dir, detectable_class + '_data',
                                                    'negatives',
                                                    file_name)
                        shutil.copyfile(src_img_path, dst_img_path)

            else:
                print('WARNING: Could not find detectable class "' + detectable_class + '". Corresponding runId: ' + run_data['runId'])

    # get runIds of recent data
    fetched_ids_path = os.path.join(target_dir, 'already_fetched.txt')
    fetched_ids = []
    if os.path.isfile(fetched_ids_path):
        with open(fetched_ids_path, 'r') as fetched_ids_file:
            fetched_ids = fetched_ids_file.readlines()
    else:
        fetched_ids = []
    fetched_ids = [fetched_id.replace('\n', '') for fetched_id in fetched_ids]

    # get, sort and fetch new data
    src_dir = os.path.join(src_dir)
    run_data_list = []
    if os.path.exists(src_dir):
        json_names = os.listdir(src_dir)
        json_names = [name for name in json_names if name[-5:] == '.json']
        for json_name in json_names:
            with open(os.path.join(src_dir, json_name), 'r') as json_file:
                run_data_list = json.load(json_file)['runData']
                if len(run_data_list) == 0:
                    raise Exception('There is no data in ' + json_name + '.')

                # sort out already fetched data
                run_data_list = [run_data for run_data in run_data_list if not run_data['runId'] in fetched_ids]

                # make sure directory already exists
                os.makedirs(os.path.join(target_dir), exist_ok=True)

                # fetch new data of this json
                with open(fetched_ids_path, 'a+') as fetched_ids_file:
                    for run_data in run_data_list:
                        Fetch(run_data)
                        fetched_ids_file.write(run_data['runId'] + '\n')
    else:
        raise Exception('The directory ' + src_dir + ' does not exist.')

def FilterOutPositivesWithTooSmallSize(min_size: int, detectable_class_id: int):
    """DEPRECATED: Ground Truth Collector forbids creation of such samples
    by this time."""
    file_name = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'pos.txt')

    lines = []
    with open(file_name, 'r') as file:

        lines = file.readlines()

    lines_to_keep = []
    for line in lines:

        bbox_info = line.split()
        bbox_info.pop(0)
        num = int(bbox_info[0])
        bbox_info.pop(0)

        no_too_small = True
        for _ in range(num):

            _, _, w, h = bbox_info[0:4]
            w = int(w)
            h = int(h)
            for _ in range(4):
                bbox_info.pop(0)

            if w < min_size or h < min_size:

                no_too_small = False
                break

        if no_too_small:

            # keep corresponding line
            lines_to_keep.append(line)

    with open(file_name.replace('.txt', '_new.txt'), 'a+') as file:

        for line in lines_to_keep:

            file.write(line)

def DetectAndShow(model_path: str,
                  imgs_dir: str,
                  image_name: str,
                  results_directory:str=None):
    """Use model_path to detect objects on the image/images image_name lying
    in imgs_dir. Save results in results_directory if given, otherwise show
    them immediately.
    
    Keyword arguments:
    model_path -- the path to the model XML file, which shall be used for
    object detection.
    imgs_dir -- the path to the directory containing the images to be
    processed.
    image_name -- either the name of an image file in imgs_dir to perform the
    detection on, or a path to a TXT file in imgs_dir, which contains a list
    of names of image files.
    results_directory -- the directory to save results in. If not given,
    results will be shown on the screen. (default: None)
    """
    
    image_paths = []
    if image_name.endswith('.txt'):
        # read in image name and build paths to them
        with open(image_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split()
                line = line[0]
                
                # build actual path to image
                image_paths.append(os.path.join(imgs_dir, line))

    else:
        image_paths = [image_name]

    # load the trained model
    classifier = cv2.CascadeClassifier(model_path)

    for path in image_paths:
        # load an image and detect objects
        img = cv2.imread(path)
        rectangles = classifier.detectMultiScale(img)
        
        # draw detection results onto original image
        for x, y, w, h in rectangles:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
        if results_directory == None:
            # display the image
            cv2.imshow('Matches', img)
            cv2.waitKey(0)
        else:
            # save the image
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            idx = None
            if platform.system().startswith('Windows'):
                idx = path.rfind('\\')
            else:
                idx = path.rfind('/')

            filename = path[idx+1:]
            filename = filename.replace('.png', '_result.png')

            cv2.imwrite(os.path.join(results_directory, filename), img)
