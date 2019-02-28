import sys
import cv2 
import numpy as np
import progressbar
from matplotlib import pyplot as plt 

def labelFrame(img,size):
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img,"{}%".format(size),(10,30),font,1,(0,255,0),2,cv2.LINE_AA)

def plotFace(img,face):
  x,y,w,h = face 
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
  video = cv2.VideoCapture(sys.argv[1])
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  sizes = np.zeros(frame_count,dtype=np.float32)

  for i in progressbar.progressbar(range(frame_count - 1)):
    ret, img = video.read()
    assert ret 
    height, width, _ = img.shape

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
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
      percent_size = 100 * (w * h) / float(height * width)  
      sizes[i] = percent_size

      labelFrame(img,percent_size)
      plotFace(img,face)
    else:
      sizes[i] = 0
      
    #cv2.imshow("img",img)
    #cv2.waitKey(25)
  
  mean_size = np.mean(sizes)
  median_size = np.median(sizes)
  max_size = np.amax(sizes)
  std_dev = np.std(sizes)
  
  plt.hist(sizes)
  plt.title("Size of Largest Face in Frame")
  plt.show()

  print("Average Size: {} \n Median Size: {} \n Max Size: {} \n Standard Deviation: {}".format(avg_size,median_size,max_size,std_dev))

if __name__ == "__main__":
  main()
