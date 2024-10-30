import cv2
from PySide6.QtCore import QThread, Signal
import numpy
import imutils
import time
from image_utils import MovingAverageQ
from backgroundRemover import removeBackground

class CameraThread(QThread):
    video_view_signal = Signal(numpy.ndarray)
    image_captured_signal = Signal(numpy.ndarray)

    def __init__(self):
        super().__init__()
        self.stop = False
        self.image_saved = False

    def start_thread(self):
        # start the thread
        self.stop = False
        self.xQ = MovingAverageQ(3)
        self.yQ = MovingAverageQ(3)
        super().start()

    def run(self):
        # run the thread
        capture = cv2.VideoCapture(1)
        self.frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        while not self.stop:
            ret, frame = capture.read()
            if ret:
                self.process_image(frame)
        capture.release()
        

    # process the image
    def process_image(self, frame):
        original_frame = frame.copy()

        #blurs the image and grabs image contours
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.blur(gray_frame, (5, 5))
        edged_frame = cv2.Canny(blurred_frame, 80, 110) 
        contours = cv2.findContours(edged_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3) 

        if len(contours) != 0:
            # calculate center of object
            x = 0
            y = 0
            validCountour = 0
            for contour in contours:

                # find the center of each contour
                center = cv2.moments(contour)

                # add center of each center of contour
                if (center["m00"] != 0):
                    centerX = int(center["m10"] / center["m00"])
                    centerY = int(center["m01"] / center["m00"])  
                    if (centerX < 0 or centerY < 0):  
                        continue
                    x += centerX
                    y += centerY
                    validCountour += 1
                    
            # calculate average of centers of contours
            if (validCountour == 0):
                return

            x = x // validCountour
            y = y // validCountour

            # draw circle at center of object
            cv2.circle(frame, (self.xQ.add(x), self.yQ.add(y)), 20, (255, 255, 255), -1)
            
            # save pic if center of object is just past the center of the frame
            if (not self.image_saved and self.xQ.get() > self.frameWidth/2):

                # determine file name
                screenshot_filename = 'sherd_' + str(time.process_time_ns()) + '.png'

                background_removed_frame = removeBackground(original_frame)
                # crop and save the image
                sherd = max(contours, key=cv2.contourArea)
                height, width, _ = original_frame.shape
                x, y, w, h = cv2.boundingRect(sherd)
                x, y, w, h = self.get_crop_area(x, y, w, h, width, height )

                cropped_frame = background_removed_frame[y:y+h,x:x+w]

                # prints out if a screenshot has been taken
                print(f"Screenshot taken: {screenshot_filename}")
                cv2.imwrite( screenshot_filename, original_frame )
                self.image_saved = True

                # remove background of image and send back to GUI for second screen
                self.image_captured_signal.emit(cropped_frame)

            if (self.xQ.get() < self.frameWidth/4):
                self.image_saved = False
   
            # sends video feed back to UI
            self.video_view_signal.emit(frame)

    # function decides how much to crop the image
    def get_crop_area(self, x,y,w,h, image_width, image_height):

        # checking top left corner
        x1 = x - 20
        if ( x1 < 0 ):
            x1 = 0
        # check top left corner
        y1 = y - 20
        if ( y1 < 0 ):
            y1 = 0
        # check width
        w1 = w + 40
        if ( w1 + x > image_width):
            w1 = image_width - x
        # check width and height
        h1 = h + 40
        if ( h1 + y > image_height):
            h1 = image_height - y
        return x1,y1,w1,h1