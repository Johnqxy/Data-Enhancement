import cv2
import json
str='save/mix3'
src=cv2.imread(str+'.jpg')
with open(str+'.json','r') as f:
    js=json.load(f)
    for i in js['shapes']:
        for j in i['points']:
            cv2.circle(src,(int(j[0]),int(j[1])),5,(255,0,0),-1)


cv2.imshow('res',src)
cv2.waitKey(0)