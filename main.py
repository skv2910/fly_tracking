import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from ultralytics import YOLO
import functions
model_path=r"C:\Users\dell\Downloads\keypt-box-model.pt"   #set model path
img_path=r"D:\summer\courtship_frames\frame_956.jpg"      #set image/frame path

img=cv.imread(img_path,cv.IMREAD_GRAYSCALE)

model=YOLO(model_path)
result=model.predict(img_path)

#formatting keys and bonding box coordinates
key=result[0].keypoints.xyn.tolist()
bbox=result[0].boxes.xywhn.tolist()
x1,y1=key[0][0]
x2,y2=key[0][1]
x3,y3=key[0][2]
x4,y4=key[0][3]
x,y,w,h=bbox[0]

x1_,y1_=key[1][0]
x2_,y2_=key[1][1]
x3_,y3_=key[1][2]
x4_,y4_=key[1][3]
x_,y_,w_,h_=bbox[1]

keypt_1=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
bbox1=[x,y,w,h]

keypt_2=[(x1_,y1_),(x2_,y2_),(x3_,y3_),(x4_,y4_)]
bbox2=[x_,y_,w_,h_]

x1=int(round(x1*img.shape[1]))
y1=int(round(y1*img.shape[0]))
x4=int(round(x4*img.shape[1]))
y4=int(round(y4*img.shape[0]))
x=int(round(x*img.shape[1]))
y=int(round(y*img.shape[0]))

x1_=int(round(x1_*img.shape[1]))
y1_=int(round(y1_*img.shape[0]))
x4_=int(round(x4_*img.shape[1]))
y4_=int(round(y4_*img.shape[0]))
x_=int(round(x_*img.shape[1]))
y_=int(round(y_*img.shape[0]))


#calculations
vector=np.array([x1-x4,y1-y4])
vector_=np.array([x1_-x4_,y1_-y4_])
dot=np.dot(vector,vector_)
magnitude1 = np.linalg.norm(vector)
magnitude2 = np.linalg.norm(vector_)
cos_angle = dot / (magnitude1 * magnitude2)
angle_rad = np.arccos(cos_angle)
angle_deg = np.degrees(angle_rad)

distance = np.linalg.norm(np.array((x,y)) - np.array((x_,y_)))
mid_pt=(int(round(x+x_)/2),int(round(y+y_)/2))

expand_factor=0.5
X1,Y1=int(round(x1+expand_factor*vector[0])),int(round(y1+expand_factor*vector[1]))
X1_,Y1_=int(round(x1_+expand_factor*vector_[0])),int(round(y1_+expand_factor*vector_[1]))

#annotating
i = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
i=cv.arrowedLine(i,(x4,y4),(X1,Y1),(255,0,0),4)
i=cv.arrowedLine(i,(x4_,y4_),(X1_,Y1_),(255,0,0),4)
i=cv.line(i,(x,y),(x_,y_),(0,0,255),4)
angle_text = f"Angle: {angle_deg:.2f} degrees"
i=cv.putText(i, angle_text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
distance_text = f"{distance:.2f}"
i=cv.putText(i, distance_text, mid_pt, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

i_rgb = cv.cvtColor(i, cv.COLOR_BGR2RGB)  

plt.imshow(i_rgb)
plt.show()






