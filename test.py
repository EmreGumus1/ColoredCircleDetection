import cv2
import numpy as np
import argparse
import sys
# hey I chanced that part



# this function is for getting values from the video itself related to its frame widht, height to be able to crate avaliable frames.
# it also returns fps to match the regular play time speed.
def parameters(videoInputName):

    if videoInputName.isOpened():
        width = videoInputName.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = videoInputName.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        width = videoInputName.get(3)  # float
        height = videoInputName.get(4)  # float

        fps = videoInputName.get(cv2.CAP_PROP_FPS)
        # This is the part for error management.
        if (width == None or height == None or fps == None):
            print("Error1: Unassined values in parameters function. \n Press Enter to quit.")
            input()
            sys.exit()
        else:
            return  int(width), int(height),fps
    else:
        return None


def findRedCircles(cimg,img):
    circles = cv2.HoughCircles(cimg[:, :, 0], cv2.HOUGH_GRADIENT, 1, cimg.shape[0] / 8, param1=10, param2=10, minRadius=1,
                               maxRadius=500)


    if circles is not None:
        circles = np.uint16(np.around(circles))[0, :]

        for i in circles:
            cv2.putText(img, 'Red Circle', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, (i[0], i[1]), i[2]+1, (0, 255, 0), 1)



def display1():
    #this is for videosi video must be in current directory.
    vcap = cv2.VideoCapture("test1.mp4")
    # this line is to capture from camera (in our case it is specified as 0). There is 2 lines to comment out for having this work properly.
    vcap = cv2.VideoCapture(0)

    # gets the given values.
    frameWidth, frameHeight, fps = parameters(vcap)#-------------Comment it out if you use the camera of the drone---------------------

    vidCheck = True
    while (vcap.isOpened() and vidCheck):
        # read the images from video.
        success, img = vcap.read()
        vidCheck = success
        if vidCheck == False:
            break

        # adopt to its frame dimentions.
        img = cv2.resize(img, (frameWidth, frameHeight))#-------------Comment it out if you use the camera of the drone---------------------

        # Converting the color space from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Generating mask to detect red color
        lower_red = np.array([0, 148, 190])#[0, 142 167, 103 190]
        upper_red = np.array([179, 255, 255])#[10, 255, 255][179, 255, 255]
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        lower_red = np.array([175, 148, 191])  # [0, 142 167, 103 190]
        upper_red = np.array([180, 255, 255])  # [10, 255, 255][179, 255, 255]
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        mask1=mask1+mask2

        # we use median blur to make the image more soft around the cirle to get better values.
        cimg = cv2.medianBlur(mask1, 5)

        # to be able to use .HoughCircles() we use gray masking
        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
        #cv2.imshow("Detected Circle", cimg[:, :, 0])
        #wait_time = int(10000)

        # Call the funtion which finds circles and show them.
        findRedCircles(cimg,img)

        cv2.imshow("Detected Circle", img)
        #cv2.imshow('original_image',mask1)
        #cv2.imshow('test', cimg[:, :, :])

        # make the image wait to match it with its regular speed.
        wait_time = int(1.000/fps)  #this can be adjusted if video is wanted to be faster
        if cv2.waitKey(wait_time) and 0xFF == ord('q'):
            break



# call the function.
display1()
