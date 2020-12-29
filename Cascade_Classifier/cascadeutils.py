import os
import cv2

def CreateNegTxt(   negs_dir_path='custom_data/negatives/',
                    negs_txt_path='custom_data/neg.txt'):

    negs_paths = os.listdir(negs_dir_path)
    negs_paths = [negs_dir_path + path for path in negs_paths]

    with open(negs_txt_path, 'a') as negTxt:
        for path in negs_paths:
            negTxt.write(path + '\n')

def CreatePosTxtAndFilterNegsOut(   class_id,
                                    poss_dir_path='custom_data/positives/',
                                    labels_dir_path='custom_data/labels/',
                                    negs_dir_path='custom_data/negatives/',
                                    poss_txt_path='custom_data/pos.txt'):

    labels_paths = os.listdir(labels_dir_path)
    labels_paths = [labels_dir_path + path for path in labels_paths]

    # iterate over all label files
    all_label_data = []
    for labels_path in labels_paths:

        with open(labels_path, 'r') as labels:

            image_path = labels_path.replace(labels_dir_path, poss_dir_path).replace('txt', 'jpg')
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
                    if int(id) == class_id:

                        x = int(round(float(x) * im_width))
                        y = int(round(float(y) * im_height))
                        w = int(round(float(w) * im_width))
                        h = int(round(float(h) * im_height))
                        line_tail.append(' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))

                except Exception as err:

                    print(err)
                    print(image_path + ' will be transfered to ' + negs_dir_path)
                    src_path = image_path
                    dest_path = image_path.replace('positives/', 'negatives/')
                    os.replace(src_path, dest_path)
                    line_tail.append('')

            line_head.append(line_tail)

        all_label_data.append(line_head)
    
    with open(poss_txt_path, 'a') as poss_txt:

        for label_file_data in all_label_data:

            if len(label_file_data[1]) > 0:
                line = label_file_data[0] + ' ' + str(len(label_file_data[1]))
                for data in label_file_data[1]:
                    line += data

                poss_txt.write(line + '\n')
            else:
                print(label_file_data[0] + ' will be transfered to ' + negs_dir_path)
                src_path = label_file_data[0]
                dest_path = label_file_data[0].replace('positives/', 'negatives/')
                os.replace(src_path, dest_path)

if __name__ == '__main__':

    CreateNegTxt()