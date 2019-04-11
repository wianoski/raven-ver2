import numpy as np
import cv2
from collections import deque
import math

class FireDetector(object):
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
                rm = 150 #220 
                #print rm
                rt = 200 #245 # 200 tdinya 225
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
                gt = 110 #100
                bt = 188 #140
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

                R1 = self.rgbfilter(image)
                R2 = self.labfilter(image)
                #R3 = rgbfilter2(image)
                R4 = self.ycrcbfilter(image)
                T = cv2.bitwise_and(R1,R2)
                B,G,R = cv2.split(image)
                div = np.divide(G,(R+1.0))
                diva = np.divide(G,(R+1.0))
                #a,b = []
                div[div>0.65] = 1
                div[div<0.25] = 1
                div[div<=0.65] = 0
                S = cv2.bitwise_and(R4,T) 
                """
                erode_kernel = np.ones((3,3),np.uint8);
                eroded_img = cv2.erode(S,erode_kernel,iterations = 1)
                    
                # dilate
                dilate_kernel = np.ones((20,20),np.uint8);
                dilate_img = cv2.dilate(eroded_img,dilate_kernel,iterations = 1)
                    
                # blog detector
                blob_params = cv2.SimpleBlobDetector_Params()
                blob_params.minDistBetweenBlobs = 10
                blob_params.filterByInertia = False
                blob_params.filterByConvexity = False
                blob_params.filterByColor = True
                blob_params.blobColor = 255
                blob_params.filterByCircularity = False
                blob_params.filterByArea = False
                #blob_params.minArea = 20
                #blob_params.maxArea = 500
                blob_detector = cv2.SimpleBlobDetector(blob_params)
                keypts = blob_detector.detect(dilate_img)
                    
                # draw centers of all keypoints in new imagez
                #blob_img = cv2.drawKeypoints(image, keypts, color=(0,255,0), flags=0)
                    
                # find largest blob
                self.detect = False   
                if len(keypts) > 0:
                        kp_max = keypts[0]
                        for kp in keypts:
                            if kp.size > kp_max.size:
                               kp_max = kp
                        cv2.circle(image,(int(kp_max.pt[0]),int(kp_max.pt[1])),int(kp_max.size),(0,255,0),2)
                        #self.count += 1

                        # set the balloon location
                        balloon_found = True
                        balloon_x = kp_max.pt[0]
                        balloon_y = kp_max.pt[1]
                        balloon_radius = kp_max.size
                """
                ############# another way ###########################
                mask = cv2.erode(S, None, iterations=2)
                mask = cv2.dilate(S, None, iterations=2)

                # find contours in the mask and initialize the current
                # (x, y) center of the ball
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None
 
	            # only proceed if at least one contour was found
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use
                    # it to compute the minimum enclosing circle and
                    # centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
        
                    #marker = cv2.minAreaRect(c)
                    #focalLength = (radius * 60.0) / 1.0 
                    #distance = (1.0 * focalLength)/ radius
                    #print 'focalLength'
                    #print focalLength
                    #print 'distance'
                    #print distance
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		            # only proceed if the radius meets a minimum size
                    if radius > 5:
			            # draw the circle and centroid on the frame,
			            # then update the list of tracked points
			            #print x,y 
                        cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                        cv2.circle(image, center, 5, (0, 0, 255), -1)
                        #img_size = math.sqrt(cam_width**2 + cam_height**2)
                        #dist1 = get_distance_from_pixels(radius, 2.0,60,img_size)
                        balloon_found = True
                        balloon_x = center[0]
                        balloon_y = center[1]
                        balloon_radius = 5
			            #rint dist1
	                    # update the points queue
	        self.pts.appendleft(center)
	        # loop over the set of tracked points
	        for i in xrange(1, len(self.pts)):
		        # if either of the tracked points are None, ignore
		        # them
                    if self.pts[i - 1] is None or self.pts[i] is None:
                        continue
 
		                # otherwise, compute the thickness of the line and
		                # draw the connecting lines
                    thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
		                #cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

                return (balloon_found, balloon_x, balloon_y, balloon_radius)

fire_finder = FireDetector()

if __name__ == "__main__":
        
        cam = cv2.VideoCapture(0)        
        detector = FireDetector()

        if cam is not None:
                while True:
                        ret, img = cam.read()
                        if(img is not None):
                                found_in_image, xpos, ypos, size = detector.filter(img)
                                print xpos,ypos
                                #show results
                                cv2.imshow('gui', img)
                                cv2.waitKey(1)
                                
                                #print ('x: {1} y: {2} size: {3}'.format(results[1],results[2],results[3]))
                        else:
                                print "failed to grab image"
