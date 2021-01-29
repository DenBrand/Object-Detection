import os
import os.path
import cv2
import math
import json
import shutil
import platform

# new classes need to be added here
DETECTABLE_CLASSES = ['cubes', 'balls', 'tetraeders']

def CreateNegTxt(detectable_class_id):

    negs_dir_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'negatives')
    negs_txt_path = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'negs.txt')

    negs_paths = os.listdir(negs_dir_path)
    negs_paths = [os.path.join(negs_dir_path, file_name) for file_name in negs_paths]

    with open(negs_txt_path, 'a') as negTxt:
        for path in negs_paths:

            path = path.replace(DETECTABLE_CLASSES[detectable_class_id] + '_data' + '\\', '') \
                        .replace(DETECTABLE_CLASSES[detectable_class_id] + '_data' + '/', '')

            negTxt.write(path + '\n')

def CreatePosTxtAndFilterNegsOut(detectable_class_id):

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
            #im_width, im_height = 320, 320
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

                    legit = True

                    # check if label makes sense
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

def FetchNewTrainingData():

    # helping function
    def FetchData(run_data):
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
                os.makedirs(os.path.join(   detectable_class + '_data', 'positives'),
                                            exist_ok=True)
                os.makedirs(os.path.join(   detectable_class + '_data', 'negatives'),
                                            exist_ok=True)

                # finally write positive entries in pos.txt
                with open(os.path.join(detectable_class + '_data', pos_txt_name), 'a+') as pos_txt:
                    for path, entries in positives_dict.items():
                        num_of_entries = len(entries)

                        line = path + ' ' + str(num_of_entries)
                        for entry in entries:
                            for key in entry.keys():
                                line += ' ' + str(entry[key])

                        # write
                        pos_txt.write(line + '\n')

                        # copy file into positives directory
                        src_img_path = os.path.join('gathered_data',
                                                    'yolo',
                                                    path.replace('positives\\', ''))
                        dst_img_path = os.path.join(detectable_class + '_data',
                                                    'positives',
                                                    path.replace('positives\\', ''))
                        shutil.copyfile(src_img_path, dst_img_path)

                # fetch negatives (write negative entries in neg.txt and move corresponding images)
                with open(os.path.join(detectable_class + '_data', neg_txt_name), 'a+') as neg_txt:
                    for path_dict in negatives:

                        path = path_dict['path']

                        # write
                        neg_txt.write(path + '\n')

                        # copy file into negatives directory
                        src_img_path = os.path.join('gathered_data',
                                                    'yolo',
                                                    path.replace('negatives\\', ''))
                        dst_img_path = os.path.join(detectable_class + '_data',
                                                    'negatives',
                                                    path.replace('negatives\\', ''))
                        shutil.copyfile(src_img_path, dst_img_path)

            else:
                print('WARNING: Could not find detectable class "' + detectable_class + '". Corresponding runId: ' + run_data['runId'])

    # get runIds of recent data
    fetched_ids_path = 'already_fetched.txt'
    fetched_ids = []
    if os.path.isfile(fetched_ids_path):
        with open(fetched_ids_path, 'r') as fetched_ids_file:
            fetched_ids = fetched_ids_file.readlines()
    else:
        fetched_ids = []
    fetched_ids = [fetched_id.replace('\n', '') for fetched_id in fetched_ids]

    # get, sort and fetch new data
    json_path = os.path.join('gathered_data', 'cascade_classifier')
    run_data_list = []
    if os.path.exists(json_path):
        json_names = os.listdir(json_path)
        json_names = [name for name in json_names if name[-5:] == '.json']
        for json_name in json_names:
            with open(os.path.join(json_path, json_name), 'r') as json_file:
                run_data_list = json.load(json_file)['runData']
                if len(run_data_list) == 0:
                    raise Exception('There is no data in ' + json_name + '.')

                # sort out already fetched data
                run_data_list = [run_data for run_data in run_data_list if not run_data['runId'] in fetched_ids]

                # fetch new data of this json
                with open(fetched_ids_path, 'a+') as fetched_ids_file:
                    for run_data in run_data_list:
                        FetchData(run_data)
                        fetched_ids_file.write(run_data['runId'] + '\n')
    else:
        raise Exception('The directory ' + json_path + ' does not exist.')

def FilterOutPositivesWithTooSmallSize(min_size, detectable_class_id):

    file_name = os.path.join(DETECTABLE_CLASSES[detectable_class_id] + '_data', 'pos.txt')

    lines = []
    with open(file_name, 'r') as file:

        lines = file.readlines()

    lines_to_keep = []
    for line in lines:

        bbox_info = line.split()
        path = bbox_info[0]
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

def CascadeDetection(detectable_class_directory, image_path, results_directory=None):

    image_paths = []
    if image_path.endswith('.txt'):
        
        with open(image_path, 'r') as file:

            lines = file.readlines()

            for line in lines:

                line = line.split()
                line = line[0] # the actual path
                
                image_paths.append(os.path.join(detectable_class_directory, line))

    else:
        image_paths = [image_path]

    # load the trained model
    classifier = cv2.CascadeClassifier(os.path.join(detectable_class_directory, 'cascade', 'cascade.xml'))

    for path in image_paths:

        # load an image
        img = cv2.imread(path)

        # detect objects
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

            idx = -1
            if platform.system().startswith('Windows'):
                idx = path.rfind('\\')
            else:
                idx = path.rfind('/')

            filename = path[idx+1:]
            filename = filename.replace('.png', '_result.png')

            cv2.imwrite(os.path.join(results_directory, filename), img)

if __name__ == '__main__':

    pass
