import cv2
import argparse
import numpy as np

# arguments for the input such pretrained and input videos
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help='path to input video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

# initialize minimum probability to eliminate weak prediction
p_min = 0.5

# threshold when applying non-maxima suppersions
thres = 0.

# VideoCapture reading video from a file
video = cv2.VideoCapture(args.video)

# preparing variable for writer
# that will use to write processed frames
writer = None

# preparing variables for spatial dimensions of the frames
h, w = None, None

# create labels into list
with open(args.classes, 'r') as f:
    labels = [line.strip() for line in f]

# Initialize colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Loading trained YOLO v3 Objects Detector
# with help of 'dnn' library form OpenCV
# Reads a network model started
network = cv2.dnn.readNetFromDarknet(args.config, args.weights)


# getting output layer names that we need from YOLO
ln = network.getLayerNames()
try:
    ln = [ln[i - 1] for i in network.getUnconnectedOutLayers()]
except:
    ln = [ln[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Defining loop for catching frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Getting dimenssions of the frame for once as everytime dimessions will be same
    if w is None or h is None:
        # Slicing and get height, width of the image
        h, w = frame.shape[:2]

    # frame processing for deep learning
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # perform a forward pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    network.setInput(blob)
    output_from_network = network.forward(ln)

    # Preparing lists for detected bounding boxes, confidences and class numbers.
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going trough all output layers after feed forward pass
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confindence_current = scores[class_current]

            if confindence_current > p_min:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data formats, we can get top left corner coordinates
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_width / 2))

                # Adding the result into prepared list
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confindence_current))
                class_numbers.append(class_current)

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence
                    
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, p_min, thres)


    # At-least one detection should exits
    if len(results) > 0:
        for i in results.flatten():
            # Getting current bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding boxes
            colour_box_current = colours[class_numbers[i]].tolist()

            # Drawing bounding box on the original image
            cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)
            
            # Prepare text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])

            # Putting txt with labeland confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    """Store processed frames into result video."""
    # Initializer writer
    if writer is None: 
        resultVideo = cv2.VideoWriter_fourcc(*'mp4v')

        # writing current processed frame into video file

        writer = cv2.VideoWriter('result-video.mp4', resultVideo, 30, 
                                 (frame.shape[1], frame.shape[0]), True)
        

    # Write processed current frame to the file
    writer.write(frame)

# Releasing video reader and writer
video.release()
writer.release()
    