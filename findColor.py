import numpy as np
import cv2
from collections import deque
import math

class Warna(object):
        def __init__(self):
                self.tedt = 1
                self.count = 0
                self.pts = deque(maxlen=32)
                
        def F(self,A, x, y):
                return np.sum((A > x) & (A < y))

        def rgbfilter(self,image):
                '''
                This is a color filter based on a method proposed in "Fire Detection Using Statistical
                Color Model in Video Sequences" by Turgay Celik, Hasan Demeril, Huseyin Ozkaramanli, Mustafa Uyguroglu
                This method uses the RGB color space and does three comparisons.
                The method returns true at any pixel that satisfies:
                red > green > blue
                red > red threshold (depends on amount of light in image)
                '''

                #b,g,r = cv2.split(image)
                #rm = cv2.mean(r)
                #rt = 220
                #R10 = cv2.compare(r,rm,cv2.CMP_GT)
                #gb = cv2.compare(g,b,cv2.CMP_GT)
                #rg = cv2.compare(r,g,cv2.CMP_GT)
                #rrt = cv2.compare(r,rt,cv2.CMP_GT)
                #rgb = cv2.bitwise_and(rrt,gb)
                #x = cv2.bitwise_and(rgb,R10)
                #A = np.divide(g,r+1.0)
                #B = np.divide(b,r+1.0)
                #C = np.divide(b,g+1.0)
                #m = self.F(A,0.25,0.65)
                #n = self.F(B,0.05,0.45)
                #o = self.F(C,0.20,0.60)
                #h = cv2.bitwise_and(x,m)
                #i = cv2.bitwise_and(h,n)

                """
                This is another way
                """
                b,g,r = cv2.split(image)
                #rm = cv2.mean(r)
                rm = 155#220 #125 #220  150
                #print rm
                rt = 224#220 #244  #245 # 200 tdinya 225
                R10 = cv2.compare(r,rm,cv2.CMP_GT)
                gb = cv2.compare(g,b,cv2.CMP_GT)
                rg = cv2.compare(r,g,cv2.CMP_GT)
                rrt = cv2.compare(r,rt,cv2.CMP_GT)
                rgb = cv2.bitwise_and(rrt,gb)
                return cv2.bitwise_and(rgb,R10)
                
                        
                #return cv2.bitwise_and(o,i)

        def rgbfilter2(self,image):
                """ 
                This is a simple threshold filter with experimental thresholds:
                r > rt (red threshold)
                g > gt (green threshold)
                b < bt (blue threshold)

                Night: rt = 0, gt = 100, bt = 140

                """

                b,g,r = cv2.split(image)
                rt = 0 #0
                gt = 130 #100 # #100 #150 110
                bt = 190#140 # #140 #140
                ggt = cv2.compare(g,gt,cv2.CMP_GT)
                bbt = cv2.compare(b,bt,cv2.CMP_LT)
                rrt = cv2.compare(r,rt,cv2.CMP_GT)
                rgb = cv2.bitwise_and(ggt,bbt)
                
                return cv2.bitwise_and(rgb,rrt)

        def labfilter(self,image):
                '''
                This is a filter based on a method proposed in "Fast and Efficient Method for Fire Detection
                Using Image Processing" by Turgay Celik

                This method uses the CIE L*a*b* color space and performs 4 bitwise filters
                The method returns true at any pixel that satisfies:
                L* > Lm* (mean of L* values)
                a* > am* (mean of a* values)
                b* > bm* (mean of b* values)
                b* > a*
                '''
                cieLab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                L,a,b = cv2.split(cieLab)
                Lm = cv2.mean(L)
                am = cv2.mean(a)
                bm = cv2.mean(b)
                R1 = cv2.compare(L,Lm,cv2.CMP_GT)
                R2 = cv2.compare(a,am,cv2.CMP_GT)
                R3 = cv2.compare(b,bm,cv2.CMP_GT)
                R4 = cv2.compare(b,a,cv2.CMP_GT)
                R12 = cv2.bitwise_and(R1,R2)
                R34 = cv2.bitwise_and(R3,R4)
                
                return cv2.bitwise_and(R12,R34)

        def ycrcbfilter(self,image):
                ycrcb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
                Y,Cr,Cb = cv2.split(ycrcb)
                ycb = cv2.compare(Y,Cb,cv2.CMP_GT)
                crcb = cv2.compare(Cr,Cb,cv2.CMP_GT)
                A1 = cv2.bitwise_and(ycb,crcb)
                ym = cv2.mean(Y)
                yym = cv2.compare(Y,ym,cv2.CMP_GT)
                A2 = cv2.bitwise_and(A1,yym)
                crm = cv2.mean(Cr)
                ccrm = cv2.compare(Cr,crm,cv2.CMP_GT)
                A3 = cv2.bitwise_and(A2,ccrm)
                cbm = cv2.mean(Cb)
                cbcbm = cv2.compare(Cb,cbm,cv2.CMP_LT)
                    
                return cv2.bitwise_and(A3,cbcbm)


        def filter(self,image):
                balloon_found = False
                balloon_x = 0
                balloon_y = 0
                balloon_radius = 0
                # Convert the HSV colorspace
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Define color range in HSV colorspace

                # Orange
                lower = np.array([5,50,50])
                upper = np.array([15,255,255])
                
                # Blue
                # lower = np.array([100,150,0])
                # upper = np.array([140,255,255])

                # Green
                sensitifity = 15
                # lower = np.array([60 - sensitifity,100,50])
                # upper = np.array([60 + sensitifity,255,255])
                
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                M = cv2.moments(mask)
                if M['m00'] > 0:
                        balloon_x = int(M['m10']/M['m00'])
                        balloon_y = int(M['m01']/M['m00'])
                        cv2.circle(image, (balloon_x, balloon_y), 20, (0,255,255), -1)
                        balloon_found = True
                # Bitwise-AND mask and original image
                res = cv2.bitwise_and(image, image, mask=mask)
                res = cv2.medianBlur(res, 5)

                return (balloon_found, balloon_x, balloon_y, balloon_radius)

color_finder = Warna()

if __name__ == "__main__":
        
        cam = cv2.VideoCapture('kocak.avi')
        # cam.get(cv2.CV_CAP_PROP_FPS, 29)
        # cam.set(CV_CAP_PROP_FPS, 29)    
        detector = Warna()
        if cam is not None:
                while True:
                        ret, img = cam.read()
                        if(img is not None):
                                found_in_image, xpos, ypos, size = detector.filter(img)
                                print xpos,ypos
                                #show results
                                # height, width = img.shape
                                # new_h=height/2
                                # new_w=width/2
                                # resize = cv2.resize(img, (new_h, new_w))
                                cv2.imshow('Kamera GG', img)
                                cv2.waitKey(1)
                                
                                #print ('x: {1} y: {2} size: {3}'.format(results[1],results[2],results[3]))
                        else:
                                print "failed to grab image"
                
