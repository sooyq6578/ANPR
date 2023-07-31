from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy.signal import find_peaks

class Morph:
    def blob_filter(self, img):
        """
        Filter out the small and large objects in the image
        :param img: Image
        """
        img_arr = np.array(img)

        # Used to find blobs where any pixels that are connected to each other, are considered a single blob
        analysis = cv2.connectedComponentsWithStats(img_arr, 4, cv2.CV_32S)
        (totalLabels, labels, stats, centroids) = analysis

        output = np.zeros(img_arr.shape, dtype="uint8")
        totalDeleted = 0
        notDeleted = []

        # Scale based on image size
        scaleHigh = img_arr.shape[1] / 1500 or 1
        low = 1000 # Min 
        high = 400000 * scaleHigh # Max
        
        notDeletedCentroid = []
        for i in range(totalLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]

            componentMask = (labels == i).astype("uint8") * 255

            # Finds irregular objects
            segmented = componentMask[y:y+h, x:x+w]
            histo = self.horizontalHistogram(segmented)
            peak, _ = find_peaks(histo, height=0)

            # Remove the small and large objects
            # Remove the objects that are too wide or too tall
            # Remove irregular objects
            if area < low or area > high or \
                (w*h > area*3.5) or \
                (w>h and ((w/h) >5.5)) or \
                (h>w and ((h/w) >5.5)) or len(peak) > 40 or \
                h*2 > img_arr.shape[0] or \
                w < 60 or h < (60 if img_arr.shape[1] > 2000 else 30):
                totalDeleted += 1
                output = cv2.bitwise_or(output, componentMask)
            else:
                # Saving the objects that survived the removal process
                notDeleted.append(i)
                notDeletedCentroid.append([x, y, w, h, [100000, 0, 100000, 0]])


        # Invert the image as it can look like the image before the blob filter
        newImage = ImageOps.invert(Image.fromarray(output))
        return notDeletedCentroid, newImage
    
    def horizontalHistogram(self, img):
        """
        Get the horizontal histogram of the image
        """
        img_arr = np.array(img)
        return np.sum(img_arr, axis=1)
    
    def closing(self, img, erosion, dilation, kernelRowSize):
        """
        Erodes the objects to remove noise, then dilates the objects to restore the size of the objects

        :param img: The image to be eroded and dilated
        :param erosion: The number of times to erode the image
        :param dilation: The number of times to dilate the image
        :param kernelRowSize: The size of the kernel to be used for horizontal dilation
        """

        # Erode the image to remove noise
        # Otsu ran again to remove erosion noises
        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(img, kernel, iterations=erosion)
        img_erosion = cv2.GaussianBlur(img_erosion, (3, 3), 0)
        ret3, img_erosion = cv2.threshold(img_erosion, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Dilate the image to restore the size of the objects
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=dilation) # Every direction
        img_dilation = cv2.dilate(img_dilation, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(kernelRowSize, 1)), iterations=dilation) # Horizontal direction

        return img_dilation