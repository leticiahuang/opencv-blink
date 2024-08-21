#learned from https://www.geeksforgeeks.org/eye-blink-detection-with-opencv-python-and-dlib/

import cv2 # for camera/video
import dlib # face & landmark (specific face features) detection
# imutils are functions for image processing, use to get eyes' landmark id
from imutils import face_utils 
from scipy.spatial import distance # calculate distance b/w eyes
from pygame import mixer # to play music

def calculate_ear(eye): 
    # calculate vertical + horizontal distances 
    y1 = distance.euclidean(eye[1], eye[5]) 
    y2 = distance.euclidean(eye[2], eye[4]) 
    x = distance.euclidean(eye[0], eye[3]) 
  
    ear = (y1+y2) / x 
    return ear 

blink_thresh = 0.3 # proportion needed to qualify as a blink
succ_blink = 1 # how may count_frames that need to be qualified < blink_thresh to count as blink
count_frame = 0 # how many continuous frames have been under blink_thresh

# get id's of eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

# set up camera and detectors
cam = cv2.VideoCapture(1) #NOTE: input should be 0 but this version of cv2 only works with 1...?
cv2.namedWindow('BlinkDetector')
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('Model/shape_predictor_68_face_landmarks.dat') 

# set up music
mixer.init()
mixer.music.load('notif.wav')

# continue to record until user hits "q"
while True:
    ret, frame = cam.read() # record a frame from video
    if not ret: # read() returns T/F, stores in ret
        print("Can't receive frame.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray for detection

    faces,_,_ = detector.run(image = frame, upsample_num_times = 0, adjust_threshold = 0.0) # detect faces in frame

    for face in faces:  # for every face in frame if multiple people 
        shape = landmark_predict(frame, face) # detect face
        shape = face_utils.shape_to_np(shape) # convert face into numpy coordinates
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 
        left_ear = calculate_ear(lefteye) 
        right_ear = calculate_ear(righteye) 

        avg = (left_ear+right_ear)/2
        if avg < blink_thresh: 
            count_frame += 1   
            if count_frame >= succ_blink: # eyes have been closed long enough to qualify as blink
                mixer.music.play()
                count_frame = 0 # reset
        else: 
            count_frame = 0
    
    cv2.imshow('BlinkDetector', frame) # display current frame on video
    if cv2.waitKey(5) & 0xFF == ord('q'): 
            break
    
cam.release()
cv2.destroyAllWindows()