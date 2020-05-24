import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import gui3
print(gui3.c)

from sort import *
from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
mystring =tk.StringVar(root)

C = Canvas(root, bg="blue", height=250, width=300)
filename = PhotoImage(file="C:\\studentproj\\counter-with-yolo-and-sort-master\\o.png")
background_label = Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

theLabel = Label(root, text="QUERY ORIENTED MULTI-VIEW DYNAMIC VEHICLE DETECTION AND TRACKING", bg="brown4", fg="white")
theLabel.pack(fill=X)

style = ttk.Style(root)
style.theme_use("clam")

frame = tk.Frame(root)
frame.pack()

bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)


lbl = Label(root,
            text="CLICK ON ENTER THE QUERY AS VEHICLE TYPE\n CLICK  -SUBMIT-  TO PREOCESS THE GIVEN INPUT\n FILE BASED ON THE QUERY, \n CLICK  -QUIT-  TO CLOSE APPLICATION",
            fg='brown4', font=("Helvetica", 18))
lbl.place(x=80, y=90)

theLabel = Label(root, text="ENTER QUERY: VEHICLE TYPE - CAR, BUS, TRUCK, MOTORBIKE, BICYCLE, HEAVY-VEHICLES ",
                 bg="brown4", fg="white")
theLabel.place(x=150, y=255)

lbl = Label(root,
            text="CLICK -RESULT- TO SEE THE OUTPUT VIDEO",
            fg='brown4', font=("Helvetica", 18))
lbl.place(x=120, y=400)

def result():
    os.system('python output.py')

def getvalue():

    # print(mystring.get())
    files = glob.glob('output/*.png')
    for f in files:
        os.remove(f)

    tracker = Sort()
    memory = {}
    line = [(53, 543), (1250, 655)]
    counter = 0
    counter_car = 0
    counter_bus = 0
    counter_truck = 0
    counter_motorbike = 0
    counter_train = 0
    counter_bicycle = 0

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True,
    #                 help="path to input video")
    # ap.add_argument("-o", "--output", required=True,
    #                 help="path to output video")
    # ap.add_argument("-y", "--yolo", required=True,
    #                 help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco", "yolov3.weights"])
    configPath = os.path.sep.join(["C:\studentproj\counter-with-yolo-and-sort-master\yolo-coco", "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vid = gui3.input_video
    vs = cv2.VideoCapture(vid)
    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        query = mystring.get()
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    if intersect(p0, p1, line[0], line[1]):
                        text = "{}".format(LABELS[classIDs[i]])
                        print("[INFO]", text)
                        counter += 1
                        if text == "car" and query == "car":
                            counter_car += 1
                            print(counter_car)
                        if text == "bicycle" and query == "bicycle":
                            counter_bicycle += 1
                        if text == "bus" and query == "bus":
                            counter_bus += 1
                        elif text == "truck" and query == "truck":
                            counter_truck += 1
                        elif text == "motorbike" and query == "motorbike":
                            counter_motorbike += 1
                        elif text == "train" and query == "train":
                            counter_train += 1

                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

            # if intersect(p0, p1, line[0], line[1]) and text == "car":
            # 	counter_car += 1
            # elif intersect(p0, p1, line[0], line[1]) and text == "motorbike":
            # 	counter_bike += 1
        # draw line
        cv2.line(frame, line[0], line[1], (255, 255, 255), 2)

        # draw counter
        # cv2.putText(frame, str(counter), (1125, 200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        # counter += 1

        # saves image file
        cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

        start_point = (20, 1200)
        start_point_in = (40, 1212)

        # Ending coordinate, here (125, 80)
        # represents the bottom right corner of rectangle
        end_point = (720, 580)
        # end_point_in = (800,500)
        # Black color in BGR
        color = (31, 14, 139)
        color_in = (255, 255, 255)

        # Black color in BGR

        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = -1
        Two_wheeler = counter_bicycle
        Two_wheeler += counter_motorbike
        Four_wheeler = counter_car
        Heavy_vehicle = counter_truck
        Heavy_vehicle += counter_bus
        Heavy_vehicle += counter_train
        # Using cv2.rectangle() method
        # Draw a rectangle of black color of thickness -1 px


        cv2.rectangle(frame, (20, 710), end_point, color, -1)
        cv2.rectangle(frame, (32, 695), (700, 600), color_in, 1)
        cv2.line(frame, (32, 645), (700, 645), color_in, 1)
        cv2.putText(frame, "QUERY ORIENTED MULTI-VIEW BOUNDING BOX REGRESSION MODEL BASED ON", (40, 617),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "REAL-TIME DYNAMIC VEHICLE DETECTION AND TRACKING", (52, 640), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1)
        if query == "car":
            cv2.putText(frame, "TOTAL CAR COUNT :", (52, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, str(counter_car), (655, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # cv2.putText(frame, "Two wheeler", (980, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, str(Two_wheeler), (1175, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, "Four wheeler", (980, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, str(Four_wheeler), (1175, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, "Heavy vehicle", (980, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, str(Heavy_vehicle), (1175, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, "Car", (980, 125), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, str(counter_car), (1175, 125), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(frame, "Bikes", (980, 150), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        elif query == "bike":
            cv2.putText(frame, "TOTAL BIKE COUNT :", (52, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, str(counter_motorbike), (655, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # cv2.putText(frame, str(counter_motorbike), (1175, 150), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)


        # cv2.putText(frame, "Trucks", (980, 175), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        elif query == "truck":
            cv2.putText(frame, "TOTAL TRUCK COUNT :", (52, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, str(counter_truck), (655, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        # cv2.putText(frame, str(counter_truck), (1175, 175), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)


        # cv2.putText(frame, "Buses", (980, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        elif query == "bus":
            cv2.putText(frame, "TOTAL BUS COUNT :", (52, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, str(counter_bus), (655, 685), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        # cv2.putText(frame, str(counter_bus), (1175, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # cv2.putText(frame, "Total Count:", (980, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
        #
        # cv2.putText(frame, str(counter), (1175, 220), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("C:\studentproj\score.avi", fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(frame)

        # increase frame index
        frameIndex += 1

        if frameIndex >= 4000:
            print("[INFO] cleaning up...")
            writer.release()
            vs.release()
            exit()

    # release the file pointers

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()






e1 = Entry(root,textvariable = mystring,width=100,fg="blue",bd=3).place(x=80, y=220)
button1 = tk.Button(root, text='Submit', fg='White',bg= 'dark green',height = 1, width = 10,command=getvalue)
button1.place(x=350, y=280)

button2 = tk.Button(bottomFrame, text="QUIT", fg="red", command=quit)

button2.pack(side=tk.BOTTOM)

button3 = Button(bottomFrame, text="RESULT", command = result)

button2.config(height=3, width=20)
button3.config(height=3, width=20)
button2.pack(side=RIGHT)
button3.pack(side=LEFT)
root.geometry("800x600+10+10")
root.mainloop()