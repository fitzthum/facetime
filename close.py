import sys
import cv2 
import numpy as np

def labelFrame(img,size):
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img,"{}%".format(size),(10,30),font,1,(0,255,0),2,cv2.LINE_AA)

def plotFace(img,face):
  x,y,w,h = face 
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
  print("img size")
  img = cv2.imread("nun.png")
  height, width, _ = img.shape

  print("Height: {}, Width: {}".format(height,width))

  # make grayscale 
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # load classifier 
  classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

  # find faces 
  faces = classifier.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=5)
  
  largest_area = 0
  largest_face = None 
  for face in faces:
    x,y,w,h = face 
    area = w*h
    if area > largest_area: 
      largest_face = face
    
  if largest_face is not None:
    x,y,w,h = face 
    percent_size = 100 * (w * h) / (height * width)  
    labelFrame(img,percent_size)
    plotFace(img,face)
  

  cv2.imshow("img",img)
  cv2.waitKey(0)

if __name__ == "__main__":
  main()
