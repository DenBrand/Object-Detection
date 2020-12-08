import cv2
import numpy as np
import argparse
import time

def load_yolo():

    net = cv2.dnn.readNet(  args.weights,
                            args.cfg)

    classes = []
    with open(args.names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return net, classes, colors, output_layers

def load_image(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=args.fx, fy=args.fy)
    height, width, channels = img.shape

    return img, height, width, channels

def detect_objects(img, net, output_layers):

    blob = cv2.dnn.blobFromImage(   img,
                                    scalefactor=1/255,
                                    size=(args.bx, args.by),
                                    mean=(0, 0, 0),
                                    swapRB=True,
                                    crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    return blob, outputs

def get_box_dimensions(outputs, height, width):

	boxes = []
	confs = []
	class_ids = []

	for output in outputs:

		for detect in output:

			scores = detect[5:]
			# print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]

			if conf > 0.3:

				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)

	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):

    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):

        if i in indexes:

            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ ==  '__main__':

    parser = argparse.ArgumentParser(   description='Use a trained dnn to perform object detection on an image',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i',
                        required=True,
                        metavar='image',
                        help='path to the image on which detection is to be done',
                        dest='img')
    parser.add_argument('-w',
                        default='darknet/weights/yolov3.weights',
                        metavar='weights',
                        help='path to the weights file',
                        dest='weights')
    parser.add_argument('-c',
                        default='darknet/cfg/yolov3.cfg',
                        metavar='config',
                        help='path to the corresponding config file',
                        dest='cfg')
    parser.add_argument('-n',
                        default='darknet/data/coco.names',
                        metavar='names',
                        help='path to the names file listing all class names',
                        dest='names')
    parser.add_argument('-fx',
                        default=0.4,
                        metavar='scale_factor_x',
                        help='rescaling factor along the x-axis',
                        dest='fx')
    parser.add_argument('-fy',
                        default=0.4,
                        metavar='scale_factor_y',
                        help='rescaling factor along the y-axis',
                        dest='fy')
    parser.add_argument('-bx',
                        default=320,
                        metavar='blob_size_x',
                        help='width of the blob data',
                        dest='bx')
    parser.add_argument('-by',
                        default=320,
                        metavar='blob_size_y',
                        help='height of the blob data',
                        dest='by')

    args = parser.parse_args()

    net, classes, colors, output_layers = load_yolo()
    img, height, width, channels = load_image(args.img)
    blob, outputs = detect_objects(img, net, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, img)