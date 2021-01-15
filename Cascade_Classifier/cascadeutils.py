import os
import os.path
#import cv2
import math
import json
import shutil

# new classes need to be added here
DETECTABLE_CLASSES = ['cubes', 'balls', 'tetraeders']

'''def CreateNegTxt(   negs_dir_path=os.path.join('custom_data', 'negatives'),
                    negs_txt_path=os.path.join('custom_data', 'neg.txt')):

    negs_paths = os.listdir(negs_dir_path)
    negs_paths = [os.path.join(negs_dir_path, path) for path in negs_paths]

    with open(negs_txt_path, 'a') as negTxt:
        for path in negs_paths:
            negTxt.write(path.replace('custom_data\\', '').replace('custom_data/', '') + '\n')

def CreatePosTxtAndFilterNegsOut(   class_id,
                                    poss_dir_path=os.path.join('custom_data', 'positives'),
                                    labels_dir_path=os.path.join('custom_data', 'labels'),
                                    negs_dir_path=os.path.join('custom_data', 'negatives'),
                                    poss_txt_path=os.path.join('custom_data', 'pos.txt')):

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
                    if  math.ceil(x + w) > im_width
                        or  math.ceil(y + h) > im_height
                        or x < 3 or x + w > im_width - 3
                        or y < 3 or y + h > im_height - 3
                        or w < 6 or h < 6:
                        legit = False

                    if id == class_id and legit:
                        
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
                os.replace(src_path, dest_path)'''

def FetchNewTrainingData():
    # TODO: Funktion muss für jede detectable Klasse zwei TXTs erstellen!!

    # helping function
    def FetchData(run_data):
        pos_txt_path = os.path.join('custom_data', 'pos.txt')
        neg_txt_path = os.path.join('custom_data', 'neg.txt')

        for detectable_class in DETECTABLE_CLASSES:
            if detectable_class in run_data.keys:

                positives = run_data[detectable_class]['positives']
                negatives = run_data[detectable_class]['negatives']

                # fetch positives
                positives_dict = {}
                for pos_entry in positives:
                    if not pos_entry['path'] in positives_dict.keys:
                        positives_dict[pos_entry['path']] = [pos_entry['boxEntry']]

                    else:
                        positives_dict[pos_entry['path']].append(pos_entry['boxEntry'])

                # Ordnerstruktur falls nicht existent für jew. Detectable Klasse erstellen und Daten DORT hin schreiben.
                # Nicht einfach in custom_data/cascade_classifier/[pos/neg].txt!!!
                # Stattdessen in custom_data/cascade_classifier/<Klassenname>_train_data/<pos/neg>.txt

                # finally write positive entries in pos.txt
                with open(pos_txt_path, 'a') as pos_txt:
                    for path, entries in positives_dict.items():
                        num_of_entries = len(entries)

                        line = path + ' ' + str(num_of_entries)
                        for entry in entries:
                            line += ' ' + str(entry)

                        pos_txt.write(line + '\n')

                        # copy file into positives directory
                        src_img_path = os.path.join('gathered_data',
                                                    'yolo',
                                                    path.replace('positives\\', ''))
                        dst_img_path = os.path.join('custom_data', path)
                        shutil.copyfile(src_img_path, dst_img_path)

                # fetch negatives (write negative entries in neg.txt and move corresponding images)
                with open(neg_txt_path, 'a') as neg_txt:
                    for _, path in negatives.items():
                        neg_txt.write(path)

                        # copy file into negatives directory
                        src_img_path = os.path.join('gathered_data',
                                                    'yolo',
                                                    path.replace('negatives\\', ''))
                        dst_img_path = os.path.join('custom_data', path)
                        shutil.copyfile(src_img_path, dst_img_path)

            else:
                print('WARNING: Could not find detectable class "' + detectable_class + '". Corresponding runId: ' + run_data['runId'])

    # get runIds of recent data
    fetched_ids_path = 'already_fetched.txt'
    run_ids = []
    fetched_ids = None
    if os.path.isfile(fetched_ids_path):
        with open(fetched_ids_path, 'r') as fetched_ids_file:
            fetched_ids = fetched_ids_file.readlines()
    else:
        fetched_ids = []

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

                # fetch new data
                with open(fetched_ids_path, 'a+') as fetched_ids_file:
                    for run_data in run_data_list:
                        FetchData(run_data)
                        fetched_ids_file.write(run_data['runId'] + '\n')
    else:
        raise Exception('The directory ' + json_path + ' does not exist.')

if __name__ == '__main__':

    FetchNewTrainingData()