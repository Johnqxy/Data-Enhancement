import cv2
import json
str='save/mix2'
src=cv2.imread(str+'.jpg')
with open(str+'.json','r') as f:
    js=json.load(f)
    for i in js['shapes'][0]['points']:
        cv2.circle(src,(int(i[0]),int(i[1])),5,(255,0,0),-1)


cv2.imshow('res',src)
cv2.waitKey(0)