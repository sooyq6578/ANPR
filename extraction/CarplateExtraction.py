from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
import cv2
import os
from Detector import Detector
from Morph import Morph
from scipy.signal import find_peaks
import math
from scipy.signal import convolve2d
from scipy.ndimage import variance

class Extraction:
    """
    The main class that extracts the car plate from the image.
    """
    def __init__(self, image, **kwargs):
        image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        self.croppedImage = self.run(image, **kwargs)

    def brightenImage(self, img) :
        """
        An original take on equalizing the brightness of an image.
        """
        img_gray = np.mean(img, axis=2)

        # Calculate the overall histogram of the image
        hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])

        # Getting the cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Calculating the mapping function from the CDF
        mapping = np.interp(np.arange(256), bins[:-1], cdf_normalized)

        # Apply the mapping function to each and every pixel in the image to obtain the an equalized image
        eq_img = mapping[img_gray.astype('uint8')]
        eq_img = Image.fromarray((eq_img * 255).astype('uint8'), mode='L')
        return eq_img

    def gaussian_blur(self, n, img, sd):
        """
        An original take on gaussian blur algorithm.
        """
        extrema_x, extrema_y = n // 2, n // 2
        filter = np.zeros((n, n), np.float32)

        for y in range(n):
            for x in range(n):
                filter[y, x] = round(self.gaussian_sop(x-extrema_x, y-extrema_y, sd), 3)

        filter = (np.rint(filter / filter[0, 0])).astype(int)

        blur_img = convolve2d(img, filter / np.sum(filter), mode="same")

        return blur_img

    def gaussian_sop(self, x, y, sd):
        """
        An original take on Gaussian Smoothing
        """
        variance = sd ** 2
        return (1/(2 * math.pi * variance) * math.e ** -((x ** 2 + y ** 2) / (2 * variance)))
    
    def cannyAlgorithm(self, img: np.array, low: int, high: int) -> np.array:
        """
        An original take on Canny edge detection algorithm.
        """
        img = Image.fromarray(img)
        Kx = np.matrix('1, 2, 1; 0, 0, 0;, -1, -2, -1')
        Ky = np.matrix('-1, 0, 1; -2, 0, 2;, -1, 0, 1')

        # Step 1: Apply Gaussian filter to smooth the image in order to remove the noise
        # blurredImage = img.filter(ImageFilter.GaussianBlur(radius=1.4))

        # Step 2a: Find the intensity gradients of the image
        # Apply the Sobel operator
        Gx = convolve2d(img, Kx, "same")
        Gy = convolve2d(img, Ky, "same")
        # # Clip range 0-255

        # Step 2b: Find the magnitude and direction of the gradients
        G = np.sqrt(Gx**2 + Gy**2)

        gDirection = np.arctan2(Gx, Gy)
        gDirectionDegree = np.degrees(gDirection) + 180
        sector = np.zeros_like(gDirectionDegree)
        # Step 2c: Calculate sector numbers of pixels according to the gradient directions
        for i in range(len(gDirectionDegree)):
            for j in range(len(gDirectionDegree[0])):
                if 0 <= gDirectionDegree[i][j] < 22.5 or 157.5 <= gDirectionDegree[i][j] <= 180 or 337.5 <= gDirectionDegree[i][j] <= 360:
                    sector[i][j] = 0
                elif 22.5 <= gDirectionDegree[i][j] < 67.5 or 202.5 <= gDirectionDegree[i][j] < 247.5:
                    sector[i][j] = 1
                elif 67.5 <= gDirectionDegree[i][j] < 112.5 or 247.5 <= gDirectionDegree[i][j] < 292.5:
                    sector[i][j] = 2
                elif 112.5 <= gDirectionDegree[i][j] < 157.5 or 292.5 <= gDirectionDegree[i][j] < 337.5:
                    sector[i][j] = 3


        # Step 3: Apply non-maximum suppression to get rid of spurious response to edge detection
        # Iterate through G
        edge = np.zeros_like(G)
        for i in range(len(G)):
            for j in range(len(G[0])):
                if sector[i][j] == 0:
                    #   |    | 
                    # x | ij | x
                    #   |    | 
                    if j == 0:
                        if G[i][j] >= G[i][j+1]:
                            edge[i][j] = G[i][j]
                    elif j == len(G[0]) - 1:
                        if G[i][j] >= G[i][j-1]:
                            edge[i][j] = G[i][j]
                    else:
                        if G[i][j] >= G[i][j-1] and G[i][j] >= G[i][j+1]:
                            edge[i][j] = G[i][j]
                elif sector[i][j] == 1:
                    #   |    | x
                    #   | ij | 
                    # x |    | 
                    firstCondition = [False, True]
                    secondCondition = [False, True]

                    try: 
                        firstCondition[0] = G[i][j] >= G[i+1][j-1]
                    except IndexError:
                        firstCondition[1] = False
                    try:
                        secondCondition[0] = G[i][j] >= G[i-1][j+1]
                    except IndexError:
                        secondCondition[1] = False

                    if firstCondition[1] and secondCondition[1]:

                        edge[i][j] = G[i][j] if firstCondition[0] and secondCondition[0] else 0
                    elif firstCondition[1]:
                        edge[i][j] = G[i][j] if firstCondition[0] else 0
                    elif secondCondition[1]:
                        edge[i][j] = G[i][j] if secondCondition[0] else 0
                elif sector[i][j] == 2:
                    #   | x  | 
                    #   | ij | 
                    #   | x  | 
                    if i == 0:
                        if G[i][j] >= G[i+1][j]:
                            edge[i][j] = G[i][j]
                    elif i == len(G) - 1:
                        if G[i][j] >= G[i-1][j]:
                            edge[i][j] = G[i][j]
                    else:
                        if G[i][j] >= G[i-1][j] and G[i][j] >= G[i+1][j]:
                            edge[i][j] = G[i][j]
                elif sector[i][j] == 3:
                    # x |    | 
                    #   | ij | 
                    #   |    | x
                    firstCondition = [False, True]
                    secondCondition = [False, True]
                    try: 
                        firstCondition[0] = G[i][j] >= G[i+1][j+1]
                    except IndexError:
                        firstCondition[1] = False
                    try:
                        secondCondition[0] = G[i][j] >= G[i-1][j-1]
                    except IndexError:
                        secondCondition[1] = False

                    if firstCondition[1] and secondCondition[1]:

                        edge[i][j] = G[i][j] if firstCondition[0] and secondCondition[0] else 0
                    elif firstCondition[1]:
                        edge[i][j] = G[i][j] if firstCondition[0] else 0
                    elif secondCondition[1]:
                        edge[i][j] = G[i][j] if secondCondition[0] else 0
                    # edge[i][j] = G[i][j] if firstCondition and secondCondition else 0

        # Step 4: Apply hysteresis threshold to get final result
        for i in range(len(gDirectionDegree)):
            for j in range(len(gDirectionDegree[0])):
                if edge[i][j] > high:
                    edge[i][j] = 255
                elif edge[i][j] < low:
                    edge[i][j] = 0
                else:
                    try:
                        if edge[i][j-1] > high or \
                            edge[i][j+1] > high or \
                            edge[i-1][j-1] > high or \
                            edge[i+1][j+1] > high or \
                            edge[i-1][j] > high or \
                            edge[i+1][j] > high or \
                            edge[i-1][j+1] > high or \
                            edge[i+1][j-1] > high:
                            edge[i][j] = 255
                        else:
                            edge[i][j] = 0
                    except IndexError:
                        edge[i][j] = 0
                    
        return edge
    

    def verticalHistogram(self, img):
        """
        Get the vertical histogram of the image
        """
        img_arr = np.array(img)
        return np.sum(img_arr, axis=0)
    
    
    def imgToGray(self, img):
        """
        Convert image to grayscale
        """
        # Check if the image is too dark, if so, equalize the image
        if variance(np.array(img)) < 2000:
            img = ImageOps.equalize(img)
            img = Image.fromarray(cv2.GaussianBlur(np.array(img), (5,5), 0))

        # Turn image into grayscale
        img_grey = img.convert("L")

        return img_grey
    
    def filterGetBestImage(self, img_grey, notDeleted):
        """
        Getting the best image from the list of images
        """
        # Getting a list of cropped images
        croppedImages = []
        for i in range(len(notDeleted)):
            croppedImages.append(img_grey.crop((notDeleted[i][4][0], notDeleted[i][4][2], notDeleted[i][4][1], notDeleted[i][4][3])))

        # Filtering the final pictures
        counter = 0
        maxCounter = -1
        maxIndex = 0
        almostPerfect = croppedImages.copy()

        # Based on the histogram, find the best candidate
        for imgs in range(len(almostPerfect)):
            histo = self.verticalHistogram(almostPerfect[imgs])
            peak, _ = find_peaks(histo, height=0)
            counter = 0
            troughPoint = 10000000
            i = 0
            increasing = 0

            # Too much peak, might fluctuate too much
            scale = almostPerfect[imgs].size[0]/500 or 1
            while i < len(peak)-1 and len(peak) < (100*scale):
                first = peak[i]
                second = peak[i+1]
                troughPoint = 10000000 # Reset the trough point

                for k in range(first, second-1):
                    troughPoint = min(troughPoint, histo[k]) # Find the trough point

                # The values of the first and second peak
                firstHisto = histo[first] 
                secondHisto = histo[second] 

                # If the trough point is less than 70% of the first or second peak, then it is a good candidate
                # If fluctuate too much, then it is a bad candidate
                if (troughPoint < firstHisto * 0.7 or troughPoint < secondHisto * 0.7) and \
                    (troughPoint > firstHisto * 0.3 or troughPoint > secondHisto * 0.3):
                    counter +=1

                # If keep increasing but no decreasing, means no black in middle on letters
                if firstHisto < secondHisto:
                    increasing += 1
                    if increasing >= 10:
                        break
                else:
                    increasing = 0
                i += 1

            # Getting the best image
            # Size must be greater than 50 and less than 1200
            # If it increases too much at once, it might be a bad candidate
            scales = almostPerfect[imgs].size[0]/500 or 1
            if counter > 0 and counter >= maxCounter and almostPerfect[imgs].size[0] > 50 and \
                almostPerfect[imgs].size[0] < 550 * scales and increasing < 2*almostPerfect[imgs].size[0]:
                maxCounter = counter
                maxIndex = imgs

        # Return none with no best image found
        if maxCounter == -1:
            print("No image found")
            return None
        

        return almostPerfect[maxIndex]


    
    def run(self, img):
        """
        Finds the best object that represents a car plate
        """
        # Step 1: Grayscale conversion
        img_grey = self.imgToGray(img)
        img_grey_arr = np.array(img_grey)

        # Step 1c: Emphasize on the edges
        getEdge = Image.fromarray(cv2.Canny(img_grey_arr, 50, 500))
        imgses = ImageChops.add(getEdge, img_grey)
        imgses_arr = np.array(imgses)

        # Step 2: Image Otsu thresholding
        blur = cv2.GaussianBlur(imgses_arr, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Step 3: Erosion and dilation
        # Step 3a: Erosion
        img_dilation = Morph().closing(th3, erosion=2, dilation=5, kernelRowSize=7)

        # Step 4: Remove large/small objects
        not_blobS = Morph().blob_filter(img_dilation)

        # Step 5: Find boxes
        try:
            image_arr, notDeleted = Detector().find_corners(img, not_blobS[1], not_blobS[0])
        except TypeError:
            print("No image found")
            return None

        # Step 6: Cropped the image into an array to be processed
        bestPlate = self.filterGetBestImage(img_grey, notDeleted)
        bestPlate.show()
        return bestPlate


if __name__ == "__main__":
    rootPath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    inputPath = os.path.join(rootPath, "Set 1 - 45/")
    # car_plate_51
    Extraction(inputPath+"000.jpg")
    # Extraction(inputPath+"001.jpg")
    # Extraction(inputPath+"002.jpg")
    # Extraction(inputPath+"003.jpg")
    # Extraction(inputPath+"004.jpg")
    # Extraction(inputPath+"005.jpg")
    # Extraction(inputPath+"006.jpg")
    # Extraction(inputPath+"007.jpg")
    # Extraction(inputPath+"008.jpg")
    # Extraction(inputPath+"009.jpg")
    # Extraction(inputPath+"010.jpg") 
    # Extraction(inputPath+"011.jpg")
    # Extraction(inputPath+"012.jpg")
    # Extraction(inputPath+"013.jpg")
    # Extraction(inputPath+"014.jpg")
    # Extraction(inputPath+"015.jpg")
    # Extraction(inputPath+"016.jpg")
    # Extraction(inputPath+"017.jpg") 
    # Extraction(inputPath+"018.jpg") 
    # Extraction(inputPath+"019.jpg") 
    # Extraction(inputPath+"020.jpg") 
    # Extraction(inputPath+"021.jpg") 
    # Extraction(inputPath+"022.jpg") 
    # Extraction(inputPath+"023.jpg") 
    # Extraction(inputPath+"024.jpg")
    # Extraction(inputPath+"025.jpg") 
    # Extraction(inputPath+"026.jpg")
    # Extraction(inputPath+"027.jpg")
    # Extraction(inputPath+"028.jpg")
    # Extraction(inputPath+"029.jpg")
    # Extraction(inputPath+"030.jpg")
    # Extraction(inputPath+"031.jpg")
    # Extraction(inputPath+"032.jpg") 
    # Extraction(inputPath+"033.jpg") 
    # Extraction(inputPath+"034.jpg")
    # Extraction(inputPath+"035.jpg")
    # Extraction(inputPath+"036.jpg")
    # Extraction(inputPath+"037.jpg")
    # Extraction(inputPath+"038.jpg")
    # Extraction(inputPath+"039.jpg")
    # Extraction(inputPath+"040.jpg")
    # Extraction(inputPath+"041.jpg")
    # Extraction(inputPath+"042.jpg")
    # Extraction(inputPath+"043.jpg")
    # Extraction(inputPath+"044.jpg")
    inputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Set 2 - 15/")
    # Extraction(inputPath+"000.jpg") 
    # Extraction(inputPath+"001.jpg")
    # Extraction(inputPath+"002.jpg")
    # Extraction(inputPath+"003.jpg")
    # Extraction(inputPath+"004.jpg")
    # Extraction(inputPath+"005.jpg")
    # Extraction(inputPath+"006.jpg")
    # Extraction(inputPath+"007.jpg")
    # Extraction(inputPath+"008.jpg")
    # Extraction(inputPath+"009.jpg")
    # Extraction(inputPath+"010.jpg")
    # Extraction(inputPath+"011.jpg")
    # Extraction(inputPath+"012.jpg")
    # Extraction(inputPath+"013.jpg")
    # Extraction(inputPath+"014.jpg")