import sys
import cv2 
import numpy as np


def main():
  print("img size")
  img = cv2.imread("nun.png")
  cv2.imshow("img",img)
  cv2.waitKey(0)

if __name__ == "__main__":
  main()
