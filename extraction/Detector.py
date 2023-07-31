import numpy as np
import cv2
class Detector:
    def hough_transform(self, img, imgEdge, notDeleted):
        """
        Finding corners based on the maximum and minimum of x and y 
        coordinates of the lines found by Hough Transform
        
        """
        lines_list = []
        image_arr = np.array(img)
        lines = cv2.HoughLinesP(np.array(imgEdge), 1, np.pi / 180, 0, minLineLength=5, maxLineGap=10)
        minX, maxX, minY, maxY = 100000, 0, 100000, 0

        # Getting midpoint of each coordinates
        for points in lines:
            x1, y1, x2, y2 = points[0]
            minX = min(minX, x1, x2)
            maxX = max(maxX, x1, x2)
            minY = min(minY, y1, y2)
            maxY = max(maxY, y1, y2)

            lines_list.append((int((x1+x2)/2),int((y1+y2)/2))) # This part usefull for finding the corners

        # Any points within the box's boundaries are considered to be part of the box
        #   notDeleted[0] = left
        #   notDeleted[1] = top
        #   notDeleted[2] = width
        #   notDeleted[3] = height
        #                      TOP
        #   (x1, y1)------------------------(x2, y1)
        #       |                               |
        #       |                               |
        #   LEFT|             MID               | LEFT + WIDTH
        #       |                               |
        #       |                               |
        #   (x1, y2)------------------------(x2, y2)
        #                  TOP + HEIGHT
        for x,y in lines_list:
            for i in range(len(notDeleted)):
                if notDeleted[i][0]-10 <= x <= notDeleted[i][0] + notDeleted[i][2]+10 and notDeleted[i][1]-10 <= y <= notDeleted[i][1] + notDeleted[i][3]+10:
                    # notDeleted[i][4].append((x, y))
                    notDeleted[i][4][0] = min(notDeleted[i][4][0], x) 
                    notDeleted[i][4][1] = max(notDeleted[i][4][1], x)
                    notDeleted[i][4][2] = min(notDeleted[i][4][2], y)
                    notDeleted[i][4][3] = max(notDeleted[i][4][3], y)

        # Fuse all nearby boxes
        notDeleted = self.fuseIntersectingBoundingBoxes(notDeleted)

        
        return image_arr, notDeleted
    
    def fuseIntersectingBoundingBoxes(self, boxes):
        """
        Bounding boxes with intersecting points will be fused together
        """
        almostFiltered = boxes.copy()
        counter = 0

        boxes = sorted(boxes, key=lambda x: x[4][0])
        boxes = sorted(boxes, key=lambda x: x[4][2])
        # Fuse all nearby boxes
        for j in range(5):
            notDone = True

            # Until it is satisfied, it will keep going through the list
            while notDone:
                notDone = False
                filtered = [almostFiltered[0]] # A new list that is going to keep all the fused boxes
                counter = 0
                for i in range(1,len(almostFiltered)):
                    # print("wow")
                    fX1 = filtered[counter][4][0] # The x1 of the last box in the filtered list
                    fX2 = filtered[counter][4][1] # The x2 of the last box in the filtered list
                    fY1 = filtered[counter][4][2] # The y1 of the last box in the filtered list
                    fY2 = filtered[counter][4][3] # The y2 of the last box in the filtered list
                    nX1 = almostFiltered[i][4][0] # The x1 of the current box in the almostFiltered list
                    nX2 = almostFiltered[i][4][1] # The x2 of the current box in the almostFiltered list
                    nY1 = almostFiltered[i][4][2] # The y1 of the current box in the almostFiltered list
                    nY2 = almostFiltered[i][4][3] # The y2 of the current box in the almostFiltered list

                    # Makes sure every corner all boxes that are intersecting are included
                    # Those that are close together are also included
                    if fX1 <= nX1 <= fX2 and fY1 <= nY1 <= fY2 or \
                        fX1 <= nX2 <= fX2 and fY1 <= nY2 <= fY2 or \
                        fX1 <= nX2 <= fX2 and fY1 <= nY1 <= fY2 or \
                        fX1 <= nX1 <= fX2 and fY1 <= nY2 <= fY2 or \
                        nX1 <= fX1 and fX1 <= nX2 and fY1 <= nY1 <= fY2 or \
                        abs(fX2 - nX1) <= 150 and abs(fY1 - nY1) <= 100:
                        filtered[counter][0] = min(filtered[counter][0], almostFiltered[i][0])
                        filtered[counter][1] = min(filtered[counter][1], almostFiltered[i][1])
                        filtered[counter][2] = filtered[counter][2] + almostFiltered[i][2]
                        filtered[counter][3] = filtered[counter][3] + almostFiltered[i][3]
                        filtered[counter][4][0] = min(filtered[counter][4][0], almostFiltered[i][4][0])
                        filtered[counter][4][1] = max(filtered[counter][4][1], almostFiltered[i][4][1])
                        filtered[counter][4][2] = min(filtered[counter][4][2], almostFiltered[i][4][2])
                        filtered[counter][4][3] = max(filtered[counter][4][3], almostFiltered[i][4][3])
                        notDone = True
                    else:
                        filtered.append(almostFiltered[i])
                        counter += 1
                if j < 4: # Sorting all possible ways, 95% chance guaranteed to select the best intersection
                    filtered = sorted(filtered, key=lambda x: x[4][j])
                almostFiltered = filtered.copy()
        return almostFiltered

    
    def find_corners(self, img, blob, notDeleted, low=50, high=200):
        """
        Finding corners based on the maximum and minimum of x and y
        coordinates of the lines found by Hough Transform
        """
        imgEdge = cv2.Canny(np.array(blob), low, high)
        return self.hough_transform(img, imgEdge, notDeleted)