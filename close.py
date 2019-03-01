import sys
import cv2 
import numpy as np
import progressbar
from matplotlib import pyplot as plt 

def labelFrame(img,size):
  size = round(size,2)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img,"{}%".format(size),(10,30),font,1,(0,255,0),2,cv2.LINE_AA)

def plotFace(img,face):
  x,y,w,h = face 
  cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
  display = False
  export = False
  step = 10
  scale = 2

  video = cv2.VideoCapture(sys.argv[1])
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  
  if export:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out.avi',fourcc,20.0,(width,height))

  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  assert frame_count > 0 
  sizes = np.zeros(frame_count/step,dtype=np.float32)

  for i in progressbar.progressbar(range(0,frame_count - 1,step)):
    if not step == 1:
      video.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret, img = video.read()
    if not ret: 
      break
    if scale > 1: 
      img = cv2.resize(img,(height/scale,width/scale))

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
      sizes[i/step] = percent_size

      if display:
        labelFrame(img,percent_size)
        plotFace(img,face)
    else:
      sizes[i/step] = 0
     
    if display:
      cv2.imshow("img",img)
      cv2.waitKey(25)
    if export: 
      out.write(img)

  video.release()
  if export:
    out.release()
  cv2.destroyAllWindows()

  mean_size = np.mean(sizes)
  median_size = np.median(sizes)
  max_size = np.amax(sizes)
  std_dev = np.std(sizes)
  
  print("Average Size: {} \n Median Size: {} \n Max Size: {} \n Standard Deviation: {}".format(mean_size,median_size,max_size,std_dev))
  
  plt.hist(sizes,bins=20)
  plt.title("Size of Largest Face in Frame")
  plt.show()


if __name__ == "__main__":
  main()
