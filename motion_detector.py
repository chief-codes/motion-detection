import cv2, pandas
from datetime import datetime

from sqlalchemy import column


first_frame = None
status_list = [None,None]
times = []
df = pandas.DataFrame(columns=["Start","End"])

video = cv2.VideoCapture(0) # start the camera
while True: 
    check, frame = video.read() # captures video frame by frame
    status=0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converts captured color frame to gray
    # it is easy to detect motion in gray objects than on color objects 
    gray = cv2.GaussianBlur(gray,(5,5),0) # this blurs the gray frame 
    #and removes image noise for more accuracy
    gray = cv2.blur(gray,(21,21))
    
    if first_frame is None: # saves first captured frame to first_frame variable 
        first_frame = gray  # the rest of the frames captured are saved gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)# finds the abs diff bet first_frame and gray 
    #and saves to delta_frame variable
    thresh_frame = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3) 
    
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())

    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("Gray frame",gray)
    cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Thresh frame", thresh_frame)
    cv2.imshow("Color frame", frame)
    
    key = cv2.waitKey(1)

    if key ==ord("q"):
        if status ==1:
            times.append(datetime.now())
        break
    
    
print(status_list)
print(times)
video.release()
cv2.destroyAllWindows()