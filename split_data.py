import argparse
import os, os.path
import random
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(   description='split data into training and test data',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s',
                        default='custom_data/images',
                        help='source image directory',
                        metavar='image source',
                        dest='imgsrc')
    parser.add_argument('-l',
                        default='custom_data/labels',
                        help='source label directory',
                        metavar='label source',
                        dest='lblsrc')
    parser.add_argument('-d',
                        default='custom_data',
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
                        help='if values in label txts should be normalized (divided by image sizes) if they aren\'t already',
                        dest='normalize')

    args = parser.parse_args()

    image_names = os.listdir(args.imgsrc)
    image_names = [file for file in image_names if file[-3:] == 'jpg' and file[-12:-4] != '_labeled']
    txt_names = os.listdir(args.lblsrc)
    txt_names = [file for file in txt_names if file[-3:] == 'txt']

    print('DELETING OLD train.txt AND test.txt ...')
    if os.path.exists(args.dst + '/train.txt'):

        os.remove(args.dst + '/train.txt')

    if os.path.exists(args.dst + '/test.txt'):

        os.remove(args.dst + '/test.txt')

    print('done.')

    print('CREATING train.txt AND test.txt ...')

    # randomly pick desired portion of data as training data
    number_training_images = (int)(len(image_names) * args.prtn / 100)
    for _ in range(number_training_images):

        random_idx = random.randrange(0, len(image_names))

        with open(args.dst + '/train.txt', 'a') as file:

            file.write('custom_data/images/' + image_names[random_idx] + '\n')

        image_names.remove(image_names[random_idx])

    print('\t' + args.dst + '/train.txt CREATED.')

    # declare the rest as test data
    with open(args.dst + '/test.txt', 'a') as file:
    
        for image in image_names:

            file.write('custom_data/images/' + image + '\n')

    print('\t' + args.dst + '/test.txt CREATED.')
    print('done.')

    # image normalization
    if args.normalize:

        print('NORMALIZING VALUES IN LABEL FILES ...')

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

        print('done.')