import time 
import numpy as np 
import cv2 
from sklearn.cluster import KMeans
from .general_utiles import spline_second_derevetave

def find_cluster_centers(image_path, num_clusters=6):
    '''
    This function takes an image path and the number of clustering centers as input, then reads the image and
    processes it by extracting the indecies of the pixels that have white or yellow color, these indecies will be passed to 
    Kmeans clustering algorithm to find the centers of these clusters, and the function will return the clustering centers.
        Parameters:
            image_path: A path of the image, str.
            num_clusters: The desired number of clusters, int.
        return:
            Array containing the centers of the clusters. 
    '''
    # Load the image
    image =cv2.imread(image_path)

    image = cv2.GaussianBlur(image, (7, 7), 0)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the white color boundaries in HSV
    lower_white = np.array([0, 0, 160], dtype=np.uint8)#v160
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    # Define lower and upper bounds for yellow color
    lower_yellow = np.array([0, 80, 100])
    upper_yellow = np.array([60, 255, 255])

    # Find white pixels within the image
    white_pixels = cv2.inRange(hsv_image, lower_white, upper_white)

    # Find yellow pixels within the image
    yellow_pixels = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Combine white and yellow pixels
    white_yellow_pixels = cv2.bitwise_or(white_pixels, yellow_pixels)

    # Find the coordinates of white and yellow pixels
    white_yellow_pixel_coords = np.argwhere(white_yellow_pixels == 255)

    # Perform K-means clustering on the white and yellow pixel coordinates
    # Adjust the number of clusters as per your requirement
    kmeans = KMeans(n_clusters=num_clusters,n_init = 10, random_state=42)
    kmeans.fit(white_yellow_pixel_coords)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_[:,::-1]
    
    return cluster_centers


def contour_drawer(img_path,control= True, upper=20, lower=1,hieght_limit=50,epsilon=0.01,number_of_inflection_points=10):
    '''
    If the Kmeans clusturing algorithm fails to find the centers of the objects of interest, this function
    attempts to draw contours around them within the image based on their colors. Also, this function could be used
    for visualizing data.
        Parameters:
            img_path: Path of the image, str.
            control: Boolean if True the approximated contours will be filtered based on some values.
            upper: The max number of points representing the approximated contours.
            lower: The min number of points representing the approximated contours.
            hieght_limit: A number specifing the peak to peak differance on the y axis.
            epsilon: A factor controlling the smoothness level of the approximated contours.
            number_of_inflection_points: A number representing the count of inflection points within the spline function.
        return
            img: A binary image with the contours drawn on it.
            approx: Array containing the approximated contours.
            total_time: The time needed for this function to be executed.       
    '''
    start= time.time()
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])#[0, 0, 180]
    upper_white = np.array([255, 25, 255])#[255, 25, 255]
    #pale_gray:35, 0, 128
    lower_gray = np.array([0, 0, 115])#60 it was used always  #[0, 0, 128] new [40, 0, 115] 115 was the best limit
    upper_gray = np.array([180, 50, 139])#50[180, 30, 200]    
    # Define lower and upper bounds for yellow color
    lower_yellow = np.array([0, 80, 100])
    upper_yellow = np.array([60, 255, 255])

    # Create a mask for white, off white,faded white colors, gray, and yellow
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply the mask to the original image
    white_img = cv2.bitwise_and(img, img, mask=mask_white)
    gray_img = cv2.bitwise_and(img, img, mask=mask_gray)
    yellow_img = cv2.bitwise_and(img, img, mask=mask_yellow)

    # Sum up the white and gray images
    result_img = cv2.add(cv2.add(white_img, gray_img), yellow_img)#cv2.add(white_img, gray_img)

    # Apply edge detection or contour detection algorithm on the extracted image
    edges = cv2.Canny(result_img, 50, 140)
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    # Approximate contours to polygons
    approx = [cv2.approxPolyDP(cnt, epsilon * cv2.arcLength(cnt, True), True) for cnt in contours]#0.01 for epsilon

    if control:
        approx=[i for i in approx if (i.shape[0]>lower) and (i.shape[0]<upper)]#5
        #check the differance between the max and min points both on the width and hight, filter out the less propable points, height=50
        approx=[contour for contour in approx if (np.ptp(contour[:,0,1])>hieght_limit) and ((np.ptp(contour[:,0,0])<450) or (np.ptp(contour[:,0,0])>35))]#50
        approx=[contour for contour in approx if spline_second_derevetave(contour) < number_of_inflection_points]

    end= time.time()
    # Draw contours on image
    cv2.drawContours(img, approx, -1, (0, 255, 0), 2)
    total_time= end - start
    return img, approx, total_time

def centers_calculator(contours):
    '''
    This function takes contour coords as input and calculates the mean of these contours;
    the means are provided as centers, which are then passed to the SAM model for filtration.
        Parameters:
            contours: Array containing coords of countours
        return:
        centers if the contours had two dim otherwise it will return 'continue' 
    '''
    try:
        centers=[np.mean(contour,axis=0) for contour in contours]
        centers=np.vstack(centers)
        return centers
    except ValueError as e:
        return 'continue'



def img_limits(centers):
    '''
    This function divides the image into regions based on the width of the image.
    The image will be divided into five parts as specified in the code,
    and the function will also determine where the polygons' centers are located.
        Parameters:
            centers: Array containing the centers of the polygons.
        return:
            A list of boolean arrays.
        '''
    limit_1=(centers[:,0]<200)
    limit_2=(centers[:,0]>200) & (centers[:,0]<850)
    limit_3=(centers[:,0]>850) & (centers[:,0]<1250)
    limit_4=(centers[:,0]>1250) & (centers[:,0]<1750)
    limit_5=(centers[:,0]>1750)
    limits= [limit_1,limit_2,limit_3,limit_4,limit_5]
    return limits    