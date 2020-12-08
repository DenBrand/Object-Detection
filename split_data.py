import argparse
import os, os.path
import random
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(   description='split data into training and test data',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s',
                        default='darknet/custom_data/images',
                        help='source image directory', 
                        metavar='source',
                        dest='src')
    parser.add_argument('-d',
                        default='darknet/custom_data',
                        help='destination directory for train.txt and test.txt',
                        metavar='destination',
                        dest='dst')
    parser.add_argument('-p',
                        default=80,
                        help='portion of training data in percent',
                        metavar='portion',
                        dest='prtn')
    parser.add_argument('-n',
                        default=False,
                        action='store_true',
                        help='if values in label txts should be normalized / divided by image sizes',
                        dest='normalize')

    args = parser.parse_args()

    file_names = os.listdir(args.src)
    image_names = [file for file in file_names if file[-3:] == 'jpg']
    txt_names = [file for file in file_names if file[-3:] == 'txt']

    print('CREATING train.txt AND test.txt')

    # randomly pick wished portion of data as training data
    number_training_images = (int)(len(image_names) * args.prtn / 100)
    for _ in range(number_training_images):

        random_idx = random.randrange(0, len(image_names))

        with open(args.dst + '/train.txt', 'a') as file:

            file.write('custom_data/images/' + image_names[random_idx] + '\n')

        image_names.remove(image_names[random_idx])

    print(args.dst + '/train.txt has been created!')

    # declare the rest as test data
    with open(args.dst + '/test.txt', 'a') as file:
    
        for image in image_names:

            file.write('custom_data/images/' + image + '\n')

    print(args.dst + '/test.txt has been created!')

    if args.normalize:

        print('NORMALIZING VALUES IN LABEL FILES')

        for txt_name in txt_names:

            # get sizes of corresponding image
            corr_img_path = args.src + '/' + txt_name[0:-3] + 'jpg'
            img = cv2.imread(corr_img_path)
            height, width, _ = img.shape

            # read original lines
            lines = None
            with open(args.src + '/' + txt_name, 'r') as file:

                lines = file.readlines()

            # infer edited lines
            new_lines = []
            for i in range(len(lines)):

                words = lines[i].split()
                new_lines.append(   words[0] + ' ' +
                                    str(int(words[1])/width) + ' ' +
                                    str(int(words[2])/height) + ' ' +
                                    str(int(words[3])/width) + ' ' +
                                    str(int(words[4])/height))

            with open(args.src + '/' + txt_name, 'w') as file:

                for line in new_lines:

                    file.write(line + '\n')

        print('Values have been normalized!')