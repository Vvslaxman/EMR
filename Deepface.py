from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
#print("OpenCV version:", cv2.__version__)

# Load the image
img1 = cv2.imread("C://Users//V.Ashok//Desktop//Laxman//me_p.jpeg")
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1_rgb[:,:,::-1])
plt.show()

res=DeepFace.analyze(img1_rgb,)#actions=['emotion'])
print(res)