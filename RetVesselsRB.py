import cv2
import numpy as np
import time
import os
from scipy.ndimage.interpolation import rotate

def adaptiveThreshold(gray, mask_dim, constant): #gray is the numpy array storing the gray-scale image after green channel extraction
    mask_dim += mask_dim%2 - 1 #to assure the mask side is odd
    ext = int(mask_dim/2) #how much to extend the image 
    image = cv2.copyMakeBorder(gray, ext, ext, ext, ext, cv2.BORDER_CONSTANT)
    for i in range(ext, image.shape[0] - ext):
        for j in range(ext, image.shape[1] - ext):
            mask = image[i-ext:i+ext, j-ext:j+ext] #pixel neighbourhood extraction 
            T_mean = np.mean(mask)-constant 
            if gray[i-ext, j-ext] > T_mean:
                gray[i-ext, j-ext] = 255
            else:
                gray[i-ext, j-ext] = 0

def noiseReduction(image, size):
    zeros = np.zeros(image.shape, dtype=np.uint8)
    n_zeros = zeros+255

    for idx, x in np.ndenumerate(image):
        if x == 0:           
            if floodFill(image, idx[0], idx[1], 0, 0, size, 1)<size: #floodfill fills the area connected to the starting pixel with a new_val = 1
                np.copyto(image, n_zeros, where = image == 1) #if the area floodFilled is smaller than size the "Ones" are substituted with 255s(background)
            else:
                np.copyto(image, zeros, where = image == 1) ##if the area floodFilled is bigger than size the "One"s are substituted with Zeros(vessel)

def floodFill(image, x, y, threshold, size, max_size, new_val, lo_diff = 0, up_diff = 0):
    
    #floodFill() is recursive: if the pixel analyzed € [threshol - lo_diff, threshold + lo_diff]
    #the recursion proceed, otherwise size is returned
    #if size reach max_size (that is noise_size) + 1 the recursion stops
    
    if size > max_size:
        return size 

    if image[x,y] >= threshold - lo_diff and image[x,y] <= threshold + up_diff:
        image[x,y] = new_val
        size += 1
        if x > 0:
            size = floodFill(image, x-1, y, threshold, size, max_size, new_val)
        if x < image.shape[0]-1:
            size = floodFill(image, x+1, y, threshold, size, max_size, new_val)
        if y > 0:
            size = floodFill(image, x, y-1, threshold, size, max_size, new_val)
        if y < image.shape[1]-1:
            size = floodFill(image, x, y+1, threshold, size, max_size, new_val)

    return size

def extractChannel(src, channel):
    colour = (channel == 'blue')*0 + (channel == 'green')*1 + (channel == 'red')*2
    img = np.zeros((src.shape[0],src.shape[1]), dtype=np.uint8)
    for idx, x in np.ndenumerate(src):
        img[idx[0],idx[1]] = src[(idx[0],idx[1],colour)]
    return img

#histogram stretching, unused in the final version of the algorithm
def imageStretching(image):
    lowerLimit = float(input("\nSet lower limit: "))
    upperLimit = float(input("\nSet upper limit: "))
    img = np.zeros(image.shape, dtype = np.uint8)
    for idx, x in np.ndenumerate(image):
        p = ((x-lowerLimit)*((255.0-0.0)/(upperLimit-lowerLimit)))+0.0
        p = int(p)
        if p < 0:
            p = 0
        if p > 255:
            p = 255
        img[idx[0], idx[1]] = p
    return img

#Grayscale inversion
def colourFlip(src):
    img = np.zeros(src.shape, dtype=np.uint8)
    for idx, x in np.ndenumerate(src):
        new_val = x*(-1)+255
        img[idx[0],idx[1]] = new_val
    return img

def generateEllipse(dim, a, b):
    set = int(dim/2)
    arr = np.zeros((dim, dim), dtype = np.uint8)

    for i in range(0, dim):
        for j in range(0, dim):
            if not((pow(i-set, 2)/pow(a,2)) + (pow(j-set, 2)/pow(b,2))) <= 1: #general ellipse equation (if a == b a circle with radius = a = b is obtained)
                arr[i,j] = 255
    
    return arr


def noiseReduction2(image):

    dim = 24
    dim += dim%2 #in case a dim is set odd
    n_zeros = np.zeros((dim, dim), dtype = np.uint8)
    n_zeros += 255
    radius = 10
    shape = generateEllipse(dim, radius, radius)
    ext = int(dim/2) 
    shapeComparison(image, shape, ext, n_zeros)

    dim = 38
    dim += dim%2 
    n_zeros = np.zeros((dim, dim), dtype = np.uint8)
    n_zeros += 255
    radius = 15
    shape = generateEllipse(dim, radius, radius)
    ext = int(dim/2)
    shapeComparison(image, shape, ext, n_zeros)
    '''The following lines are left commented because they were too aggressive with some images
    so they are unused when the algorithm is performed on all the twenty images'''
    # a = 4
    # b = 15
    # shape = generateEllipse(dim, a, b)
    # for i in range(0,4):
    #     n_zeros = np.zeros((dim, dim), dtype = np.uint8)
    #     n_zeros += 255
    #     shape = rotate(shape, i*20, order=0, reshape=False, mode='nearest')
    #     shapeComparison(image, shape, ext, n_zeros, 0.9) 

def shapeComparison(image, shape, ext, n_zeros, tolerance=0.7):
    n_zeros = colourFlip(shape) #values to substitute in case of high correspondence between mask and the shape
    for i in range(ext+5, image.shape[0]-ext-5):
        for j in range(ext+5, image.shape[1]-ext-5):
            mask = image[i-ext:i+ext, j-ext:j+ext] #extraction of the pixel neighbourhood
            if np.sum(mask == shape)/pow(ext*2,2) > tolerance:
                np.copyto(image[i-ext:i+ext, j-ext:j+ext], n_zeros, where = n_zeros == 255) #if there is correspondance between the mask and the shape the area is set to background
    

def main(path):
    start = time.time()

    #insert here the path where to save the output
    destination = r'C:\Users\raffo\OneDrive - Politecnico di Milano\Codici\Python\ImageProcessing\Immagini\tmp5'

    name = path[-10:-4]
    print(name)
    mask_dim = 21 #dimension of the pixel neighbourhood for the threshold calculation
    constant = 2 #constant subtracted after the mean of the pixel neighbourhood is calculated
    noise_size = 100 #minimum dimension that a certain area labeled as vessel has to have in order to not be re-labeled as background
    
    img = cv2.imread(path)

    image = extractChannel(img, "green") #"blue", "green", "red"
    
    #uncomment to save the green channel
    #cv2.imwrite(os.path.join(destination,name + ".png"), image)

    adaptiveThreshold(image, mask_dim, constant)

    image = colourFlip(image)
    #uncomment to save the image before noise reduction
    #cv2.imwrite(os.path.join(destination,name + "_" + str(mask_dim) + "-" + str(constant) + "-" + "noisy"+ ".png"), image)
    image = colourFlip(image)

    print("\nAdaptive threshold applied, proceeding with noise reduction...")

    noiseReduction(image, noise_size) #floodFill technique
    noiseReduction2(image) #elliptical shapes comparison
    noiseReduction(image, 200) #floodFill technique

    image = colourFlip(image) #all the algorithm has worked with vessels set to 0 and background to 255

    filename = os.path.join(destination,name + "_" + str(mask_dim) + "-" + str(constant) + "-" + str(noise_size) + ".png")

    cv2.imwrite(filename, image)

    end = time.time()

    print("\nExecution in " + str(end-start) + " seconds\n")

if __name__ == "__main__":
    
    #insert here the path where the original images(.ppm format) are stored
    folderPath = r'C:\Users\raffo\OneDrive - Politecnico di Milano\Codici\Python\ImageProcessing\Immagini\Originalippm'
    
    # for filename in os.listdir(folderPath):
    #     if filename[-3:len(filename)]=="ppm":
    #        main(os.path.join(folderPath,filename))

    '''uncomment the for loop and comment the following two lines to execute 
    the algorithm on all the .ppm images present in folderPath'''

    filename = 'im0162.ppm'
    main(os.path.join(folderPath,filename))

    
    
