import cv2

#our image
img_file = 'car.jpg'
video = cv2.VideoCapture('testvideo.mp4')

#pre trained car and pedestrian classifier
#car_tracker_file = 'AnyConv.com__1car.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#create car and pedestrian classifier
#car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#run forever
while True:
    #read current frame
    (read_successful, frame) = video.read()
    #safe coding
    if read_successful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

#detect cars and pedestrian
#cars = car_tracker.detectMultiScale(grayscaled_frame)
pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

#draw rectangles around cars
#for (x, y, w, h) in cars:
    #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#draw rectangles around pedestrians
for (x, y, w, h) in pedestrians:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

#display the  image
cv2.imshow('my pedestrian and car detector', frame)

#dont autoclose wait for for key press
cv2.waitKey()

#release the video capture object
#video.release()

#shows that everything is running
print('code completed')

#stop if q key is pressed
#if key==81 or key==113:
   # break
