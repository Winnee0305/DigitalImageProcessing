#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:05:37 2024

@author: Chew Win Nee, Chia Pei Qin
"""

from matplotlib import pyplot as pt
import numpy as np
import cv2
import sys

def processVideo(videoFileName):
    out= cv2.VideoWriter("processed_%s.avi"%(videoFileName), cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280,720)) # modify the output video properties
    processContent(out, videoFileName)
    processEndScreen(out)
    print("Done video processing")
    
def processContent(out, videoFileName):
    vid, vid_faster, vid_talking, vid_total_frames, talking_total_frames = initializeVideo(videoFileName)    
    temp_frame_lst = []          # temporary store 3 frames information, X-1, X, X+1 (to be used in face detection function)
    narrator_frame_count = 0     #taling video frame count
    
    logo = createLogo()
    gradient = createGradient(64, 64, np.array([255, 182, 193]), np.array([255, 105, 180])) #(lighter pink): RGB (255, 182, 193), (darker pink): RGB (255, 105, 180)
    watermarks = [readImage("watermark1"), readImage("watermark2")]                         # Read watermarks
    
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    isNotSameSize = False
    if (width != 1280 and height != 720):
        isNotSameSize = True
    
    '''Video Processing'''
    
    for frame_count in range(int(vid_total_frames)):              # process the frame one by one (1s -> 30 frames)
        print("Processing %s video content: %d / %d"%(videoFileName, frame_count, vid_total_frames))
        success, frame = vid.read()  # read a frame from the original video
        
        '''Adjust Brightness'''
        output = adjustBrightness(frame)
         
        '''Face Blurring'''
        faces_current_frame = faceDetection(frame)                      # detect faces in current frame
        updateTempLst(temp_frame_lst, faces_current_frame)              # update the temp lst
   
        if frame_count+1 != vid_total_frames and frame_count != 0 :     # if this is not the first or last frame
            success, next_frame = vid_faster.read()                     # read the next frame 
            faces_next_frame = faceDetection(next_frame)                # detect the faces in the next frame
            updateTempLst(temp_frame_lst, faces_next_frame)             # store the information of next frame
            temp_frame_lst = secondaryFaceDetection(temp_frame_lst)     # if there is have three frames information, apply secondary face detection function
            faces_current_frame = temp_frame_lst[1]                     # update the faces current frame after fixed in the secondary face detection   
       
        output = faceBlurring(output, faces_current_frame)
        
        '''Adjust Size'''
        if isNotSameSize:
            output = resize(output, [720, 1280], [720, 1280], "Whole", False)
        
        '''For Talking video'''
        
        if narrator_frame_count < talking_total_frames:
            output = addNarrator(vid_talking , output, narrator_frame_count)
            narrator_frame_count += 2                   # To skip the alternate frame -> reduce fps (because talking video is 60 fps while output video is 30 fps)

        '''Adding Logo'''  
        output = addLogo(logo, gradient, output)  
    
        
        '''Adding Watermark'''
        output = addWatermark(watermarks, output, frame_count)
        
        '''Adding Fade Effect'''
        output = applyFadeEffect(output, frame_count, vid_total_frames, 5, "Fade In")
        output = applyFadeEffect(output, frame_count, vid_total_frames, 5, "Fade Out")
        
        out.write(output)
    
def processEndScreen(out):
    end_vid = readVideo("endscreen")
    end_total_frames = getTotalFrame(end_vid)
    for end_frame_count in range(int(end_total_frames)):
        print("Processing end screen: %d / %d"%(end_frame_count, end_total_frames))
        success, end_frame = end_vid.read()
        output = applyFadeEffect(end_frame, end_frame_count, end_total_frames, 3, "Fade In")
        output = applyFadeEffect(output, end_frame_count, end_total_frames, 3, "Fade Out")
        out.write(output)
        
def readVideo(fileName):
    vid = cv2.VideoCapture("%s.mp4"%(fileName))
    return vid

def readImage(fileName):
    img = cv2.imread("%s.png"%(fileName),1)
    return img

def getTotalFrame(vid):
    total_frame = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    return total_frame

def bgr2gray(img):
    output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return output

def gray2bgr(img):
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return output

def bgr2rgb(img):
    output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return output

def initializeVideo(videoFileName):
    vid = readVideo(videoFileName)                       # read the original video
    vid_faster = readVideo(videoFileName)                # copy a video (to be used in face detection)
    vid_faster.set(1, 1)                                 # in the copied video - set to the next frame
    vid_talking = readVideo("talking")                   # read the talking video
    vid_total_frames = getTotalFrame(vid)                # count the number of frames of the original video
    talking_total_frames = getTotalFrame(vid_talking)    # count the number of frames of talking video
    
    return vid, vid_faster, vid_talking, vid_total_frames, talking_total_frames
    
''' Funtion 1. Daytime or Nighttime Detection & Adjust Brightness ''' 
    
def adjustBrightness(img):
    gray_img= bgr2gray(img)
    mean_intensity = np.mean(gray_img)
    if (mean_intensity < 100) :             # Mean intensity that less than 100 is considered as night time
        diff = 110 - mean_intensity
        img = cv2.addWeighted(img, 1, img, 0 , diff)
    return img
    
''' Function 2. Face Detection & Blurring'''
    
def updateTempLst(lst, lstToAdd):
    if len(lst) >= 3 :      # If the temp lst have already stored information of three frames, apply FIFO
        lst.pop(0)
    lst.append(lstToAdd)

def faceDetection(img):
    face_cascade = cv2.CascadeClassifier("face_detector.xml")
    facesLst = face_cascade.detectMultiScale(img, 1.3, 5)
    return facesLst

def secondaryFaceDetection(lst):
    checkedLst = [list(sublist) for sublist in lst]
    standard_diff = 250                              # define a standard tolerance (the size of area around to detect the same face)
    
    for face_previous in lst[0]:                     # check the faces occur in frame X-1 one by one
        x_previous, y_previous, w, h = face_previous
        
        for face_forward in lst[-1]:                 # check the faces occur in frame X+1 one by one
            x_forward, y_forward = face_forward[:2]
            x_diff = abs(x_previous - x_forward)     # check the position of face at previous and forward
            y_diff = abs(y_previous - y_forward)
            
            if 0 <= x_diff <= standard_diff and 0 <= y_diff <= standard_diff:    # same face is detected in the previous and forward frame
                face_fail_to_detect = True
                for face_current in lst[1]:          # check the faces occur in current frame one by one
                    x_current, y_current = face_current[:2]
                    # face is detected around the location as previous frame and after frame  
                    if (x_previous-standard_diff<= x_current <= x_previous+standard_diff) and (y_forward-standard_diff <= y_current <= y_forward+standard_diff):  
                        face_fail_to_detect = False  # indicates face are successfully detected
            
                if face_fail_to_detect:   # if face is fail to detect
                    ## manually add the face loacation to the current frame based on mean coordinate of previous and forward frames       
                    faceDetected = [(x_previous + x_forward) // 2, (y_previous + y_forward) // 2, w, h] 
                    checkedLst[1].append(faceDetected) # update the faces lst
    return checkedLst
        
def faceBlurring(img, faces):
    processed_img = img.copy()
    for (x, y, w, h) in faces:
        face_patch = img[y:y+h, x:x+w]                                                 # Declare the area of face detected
        blurred_patch = cv2.blur(face_patch, (30, 30), borderType=cv2.BORDER_DEFAULT)  # Using Open CV blur function to blur the face patch 
        blurred_patch_resized = cv2.resize(blurred_patch, (face_patch.shape[1], face_patch.shape[0]))    # Resize the blurred_patch (w, h) values
        processed_img[y:y+h, x:x+w] = blurred_patch_resized                # Replace the same segment in the original img with the processed patch
    return processed_img

'''Function 3. Add Element (narrator, logo, watermark)'''

def addNarrator(vid_narrator, frame, narrator_frame_count):
    narrator = getSpecificFrame(vid_narrator, narrator_frame_count)
    narrator_mask = extractMaskByPeakColor(narrator)
    resized_mask = resize(narrator_mask, frame.shape[:2], [288,512], "BottomLeft", True)
    resized_narrator = resize(narrator, frame.shape[:2], [288,512], "BottomLeft", False)
    output = mergeBackgroundnForeground(frame, resized_mask, resized_narrator, False)
    return output

def resize(frame, bg_size, target_size, position, grayStatus):
    height, width = bg_size[:]
    if position == "BottomLeft":
        start_y = height - target_size[0]
        end_y = height
        start_x = 0
        end_x = target_size[1]
    if position == "TopRight": 
        margin = 200
        start_y = margin
        end_y = target_size[0] + margin 
        start_x = width - target_size[1] - margin
        end_x = width - margin
    if position == "Whole":
        start_y = 0
        end_y = target_size[0]
        start_x = 0
        end_x = target_size[1]
    resized_fg = cv2.resize(frame, (target_size[1], target_size[0]))
    output = np.zeros((height, width,3), dtype=np.uint8)
    
    if grayStatus: # Convert to bgr if it is a binary image (to avoid shape mismatch in future operation)
        resized_fg = gray2bgr(resized_fg)
    
    output[start_y:end_y, start_x:end_x] = resized_fg # Place the foreground image at the correct position of black image
    return output

def mergeBackgroundnForeground(bg, mask, fg, grayStatus):
    output = bg.copy
    inv_mask = cv2.bitwise_not(mask)
    
    if grayStatus: # Gray status -> indicates the mask is binary (will need to convert to bgr)
        inv_mask = gray2bgr(inv_mask)
        mask = gray2bgr(mask)

    bg_masked = cv2.bitwise_and(bg, inv_mask)  # Make a hole at the background image
    fg_masked = cv2.bitwise_and(fg, mask)
    output = cv2.add(bg_masked, fg_masked)
    return output

def getSpecificFrame(vid, frame_count):
    vid.set(1, frame_count)     # Set to the respective frame of talking video
    success, frame= vid.read()  # Read a frame from the talking video
    return frame
    
def extractMaskByPeakColor(img): 
    peakColors = calcPeakBGR(img)
    peak_bgr = np.array([[peakColors]], dtype=np.uint8)
    peak_hsv = cv2.cvtColor(peak_bgr, cv2.COLOR_BGR2HSV)[0][0]

    h_margin= 30        # A tolerance diff to allow in the range
    s_margin= 130
    v_margin = 160
    
    '''
    Open CV HSV Range:  H - 0 to 179
                        S - 0 to 255
                        V - 0 to 255
    '''

    h_low = max(0, int(peak_hsv[0]) - h_margin)     # Define the lower and upper value for each hsv
    s_low = max(0, int(peak_hsv[1]) - s_margin)
    v_low = max(0, int(peak_hsv[2]) - v_margin)
    
    h_up = min(179, int(peak_hsv[0]) + h_margin)
    s_up = min(255, int(peak_hsv[1]) + s_margin)
    v_up = min(255, int(peak_hsv[2]) + v_margin)
    
    lower_boundary = np.array([h_low, s_low, v_low])
    upper_boundary = np.array([h_up, s_up, v_up])
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inverted_mask = cv2.inRange(img_hsv, lower_boundary, upper_boundary)    # Highlight the area in this color range (in the color range -> white)    
    mask = cv2.bitwise_not(inverted_mask)
    return mask

def extractMaskByThreshold(img, threshold):
    [nrow, ncol] = img.shape
    output = np.zeros([nrow, ncol], dtype=np.uint8)
    for x in range(nrow):
        for y in range(ncol):
            if img[x,y] < threshold:
                output[x,y] = 1
    return output

def calcPeakBGR(img):
    b, g, r = cv2.split(img)
    hist_b = cv2.calcHist([b], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0,256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0,256])
    peak_b = np.argmax(hist_b)
    peak_g = np.argmax(hist_g)
    peak_r = np.argmax(hist_r)
    return [peak_b, peak_g, peak_r]

def createLogo():
    logo_arrays = np.full((64, 64), 0, dtype=np.uint8)
    rectangles = [
        (3, 6, 11, 23), (3, 6, 36, 47), (6, 8, 8, 12), (6, 8, 22, 25), 
        (6, 8, 47, 50), (6, 8, 33, 36), (8, 9, 6, 12), (8, 9, 22, 25), 
        (8, 9, 33, 36), (8, 9, 47, 50), (9, 11, 6, 9), (9, 11, 14, 20), 
        (9, 11, 25, 28), (9, 11, 31, 33), (9, 11, 39, 44), (9, 11, 50, 52), 
        (11, 22, 3, 6), (11, 22, 28, 31), (11, 22, 52, 55), (11, 14, 11, 22), 
        (11, 14, 36, 47), (14, 20, 9, 14), (14, 20, 20, 25), (14, 20, 34, 39), 
        (14, 20, 44, 50), (20, 22, 11, 22), (20, 22, 36, 47), (22, 25, 6, 9), 
        (22, 25, 14, 20), (22, 25, 25, 28), (22, 25, 31, 33), (22, 25, 39, 45), 
        (22, 25, 50, 53), (25, 28, 9, 11), (25, 28, 22, 36), (25, 28, 47, 50), 
        (28, 31, 6, 9), (28, 31, 11, 22), (28, 31, 36, 47), (28, 31, 55, 61), 
        (31, 55, 3, 6), (31, 55, 47, 50), (31, 58, 58, 61), (36, 39, 6, 25), 
        (31, 34, 52, 55), (33, 36, 50, 53), (39, 47, 25, 28), (47, 50, 6, 25), 
        (47, 53, 36, 42), (50, 53, 50, 53), (52, 55, 52, 55), (55, 58, 6, 9), 
        (55, 58, 44, 47), (55, 58, 55, 58), (58, 61, 9, 44)
    ]    # Define the list of black rectangle coordinates
    for (y1, y2, x1, x2) in rectangles:         # Apply the black rectangles to the logo array
        logo_arrays[y1:y2, x1:x2] = 255
    logo_rgb = cv2.cvtColor(logo_arrays, cv2.COLOR_GRAY2RGB)
    return logo_rgb

def createGradient(height, width, start_color, end_color):
    gradient = np.linspace(start_color, end_color, width, dtype=np.uint8)  # Create a linear gradient
    gradient_image = np.tile(gradient, (height, 1, 1))   # Repeat the gradient to fill the image
    return gradient_image

def addLogo(logo, gradient, frame):
    resized_mask = resize(logo, frame.shape[:2], [64,64], "TopRight", False)
    resized_gradient = resize(gradient, frame.shape[:2], [64,64], "TopRight", False)
    output = mergeBackgroundnForeground(frame, resized_mask, resized_gradient, False)
    return output

def addWatermark(watermarks, frame, frame_count):
    current_watermark = (frame_count // (30 * 5)) % len(watermarks)     #Identify which watermark to use based on the frame number
    watermark = watermarks[current_watermark]
    output = cv2.add(frame, watermark)
    return output

def applyFadeEffect(frame, frame_count, vid_total_frames, fade_time, effect):
    total_fade_frames = fade_time * 30      # Number of frames to fade in (5 seconds x 30 fps)
    if effect == "Fade In":
        fade_frame_count = frame_count
    elif effect == "Fade Out":
        fade_frame_count = vid_total_frames - frame_count

    if fade_frame_count < (total_fade_frames):
        effect_factor = fade_frame_count / (total_fade_frames - 1)
        frame = (frame * effect_factor).astype(np.uint8)
    return frame

'''Main Function'''

def main():
    opt = ""
    while opt.upper() != "Q":
        print()
        print("Enter [1] to process 'street' video")
        print("Enter [2] to process 'traffic' video")
        print("Enter [3] to process 'singapore' video")
        print("Enter [4] to process 'office' video")
        print("Enter [Q] to exit the program")
        print("-" * 60)
        opt = input("Enter option >> ")
        if opt == "1":
            processVideo("street")
        elif opt == "2":
            processVideo("traffic")
        elif opt == "3":
            processVideo("singapore")
        elif opt == "4":
            processVideo("office")
        elif opt.upper() == "Q":
            sys.exit()
        else:
            print("Invalid option")
            
if __name__ == "__main__":
    main()