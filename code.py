import math
import numpy as np
import pandas as pd
import skimage.io as skio
from skimage import measure
from skimage import morphology
import matplotlib.pyplot as plt
import skimage.filters as skFilters
from skimage.transform import rotate
from skimage.segmentation import clear_border

def getCanny(image):
    im = skFilters.sobel(image)
    im = skFilters.gaussian(im)
    return im

def getThresholdedImage(image):
    r = skFilters.threshold_otsu(image)
    binary_image = 1*(image > r)
    return binary_image

chawal_type = 'super kernel naya'

chawal = []
chawal_number = 0
#Arrays
rice_name = []
rice_type = []
generalColor = []
lengthInMM = []
widthInMM = []
#Loop ranging from 1 to 10 because there are 10 pictures for each type of rice
for c in range (1,11):
    chawal.append('sk'+str(c))
    image = skio.imread(chawal[c-1]+'.jpeg', as_gray=True);
    
    
    # structuring element
    se  = np.ones(shape=(7,7), dtype=np.uint8)
    
    #opening the image
    image = morphology.opening(image,se)
    
    # Thresholded Image
    thresholded = getThresholdedImage(image)
    # thresholded = morphology.closing(thresholded, se)
    
    # removing objects that are touching the border
    edge_touching_removed = clear_border(thresholded)
    # plt.imshow(thresholded, cmap='gray')
    
    # Labeling regions
    labeled_image = measure.label(edge_touching_removed, connectivity=image.ndim)
    
    #Region props
    props = measure.regionprops_table(labeled_image, image,
                                      properties = ['label',
                                                    'area',
                                                    'bbox',
                                                    'major_axis_length',
                                                    'minor_axis_length',
                                                    'orientation',
                                                    'equivalent_diameter'
                                                    ])
    
    df = pd.DataFrame(props)
    angles = df['orientation']
    #Converting dataframe(angles) to list so that it can be processed in loop
    angles_degree = list((angles))
    # size = number of rows in dataframe(df)
    size = df.shape[0]
    
    # angles_degree = np.array(angles)
    
    #Creating a list with angles in degrees
    for i in range(0,size):
          angles_degree[i] = math.degrees(angles_degree[i])
    
    # Adding the angles_degree array in dataframe
    df['angle_degrees'] = angles_degree
    
    # Converting the dataframe to numpy array so that it can be processed in loop
    objects = df.to_numpy()
    
    # assigning the indexes of different properties in numpy array to variables
    label = 0
    area = 1
    bbox0 = 2
    bbox1 = 3
    bbox2 = 4
    bbox3 = 5
    majorAxis = 6
    minorAxis = 7
    orientation = 8
    diameter = 9
    angleDegree = 10
    
    # Area of coin
    maxArea = np.max(objects[:, area])
    
    # Declaring arrays
    # chawalKiMajorPitch = []
    # chawalKiMinorPitch = []
    
    # declaring variable for finding index at which coin is present
    indexOfCoin = 0
    
    # Diamaeter in mm of 5 rupee coin
    coinDiameterMM = 19
    
    # rows and cols for total objects
    row,col = objects.shape
    
    # Finding index at which coins is present in objects array
    for r in range(0, row):
        checkArea = objects[r, area]
        if checkArea == maxArea:
            indexOfCoin = r
            break
        
    # Array that contains all properties of coins
    coinArray = objects[indexOfCoin]
    
    # Finding Pixel Pitch of coin
    coinPixelPitch = coinArray[diameter]/coinDiameterMM
    
    # Finding length of one pixel
    aikPixelKiLength = 1/coinPixelPitch
    

    for r in range(0, row):
        #area of the object
        checkArea = objects[r, area]
        # If not coin
        if(checkArea < maxArea):
            
            b0 = int(objects[r, bbox0]) 
            b1 = int(objects[r, bbox1]) 
            b2 = int(objects[r, bbox2])
            b3 = int(objects[r, bbox3])
            
            angle = objects[r, angleDegree]
            
            lengt = objects[r, majorAxis] * aikPixelKiLength
            lengthInMM.append(lengt)
            
            widt = objects[r, minorAxis] * aikPixelKiLength
            widthInMM.append(widt)
            if(r < row // 2):
                rice_name.append('F - Rice '+ str(chawal_number))
            else:
                rice_name.append('A - Rice '+ str(chawal_number))
            chawal_number = chawal_number + 1
            rice_type.append(chawal_type)
            
            generalColor.append('White')
            
            cropped = edge_touching_removed[b0:b2, b1:b3]
            colorCrop = image[b0:b2 , b1:b3]
    
            cropped = rotate(cropped, 360 - angle, resize = True)
            cropped = skFilters.median(cropped, np.ones(shape=(5,5)))
            
            colorCrop = rotate(colorCrop, 360 - angle, resize = True)
            colorCrop = skFilters.median(colorCrop, np.ones(shape=(5,5)))
                   
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(cropped, cmap='gray')
            plt.title("Thresholded")
            plt.subplot(1,2,2)
            plt.imshow(colorCrop, cmap='gray')
            plt.title("Orignal")
            plt.imsave('./cropped '+chawal_type+'/'+rice_name[chawal_number - 1]+'.jpeg', colorCrop, cmap='gray')
            plt.axis("off")
    
riceDict = dict({'Rice Name' : rice_name,
                'Rice Type' : rice_type ,
                'General Color' : generalColor,
                'Length in mm' : lengthInMM,
                'Width in mm' : widthInMM
                })
    
riceDF = pd.DataFrame(riceDict)

riceDF.to_csv(chawal_type + '.csv', index = False)


    
    


