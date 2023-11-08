import numpy as np 
import cv2 as cv

classes=["BACKGROUND","AEROPLANE","BICYCLE","BIRD","BOAT","BOTTLE","BUS","CAR","CAT","CHAIR","COW", 
		"DININGTABLE","DOG","HORSE","MOTORBIKE","PERSON","POTTEDPLANT","SHEEP","SOFA","TRAIN","TVMONITOR"]
colors=np.random.uniform(0,100,size=(len(classes),3))
net=cv.dnn.readNetFromCaffe("MobileNetSSD.prototxt","MobileNetSSD.caffemodel")

capt=cv.VideoCapture("VDO/cat&dog.mp4")

while True:
	ret,frame=capt.read()
	if ret:
		(h,w)=frame.shape[:2]
		blob=cv.dnn.blobFromImage(frame,0.007843,(300,300),127.5)
		net.setInput(blob)
		detections=net.forward()

		for i in np.arange(0, detections.shape[2]):
			percent=detections[0,0,i,2]
			if percent>0.8:
				class_index=int(detections[0,0,i,1])
				box=detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startX,startY,endX,endY)=box.astype("int")

				label="{} [{:.2f}%]".format(classes[class_index],percent*100)
				cv.rectangle(frame,(startX,startY),(endX,endY),colors[class_index],2)
				cv.rectangle(frame,(startX-1,startY-30),(endX+1,startY),colors[class_index],cv.FILLED)
				make=startY-15 if startY-15>15 else startY+15
				cv.putText(frame,label,(startX+20, make+5), cv.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),1)

		cv.imshow("Frame",frame)
		if cv.waitKey(1)&0xFF==ord("e"):
			break
		
capt.release()
cv.destroyAllWindows()