import os
import glob
import json
from pathlib import Path
import numpy as np 
import torch
import cv2 
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt 
from segment_anything import sam_model_registry, SamPredictor 
import yaml

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def model_loader_predictor(model_type = "vit_h",sam_checkpoint =r"C:\Users\matou\Downloads\sam_vit_h_4b8939.pth",device=device):
    '''
        Loads SAM model to the available device (GPU/CPU).
        Parameters:
            sam_checkpoint: path to SAM weights.
        return:
          SAM model
            '''
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor

def images_paths(images_directory):
    return glob.glob(images_directory+ '/*.png')


def json_file_data_format(img_path):
    '''This function generates a dictionary in the labelme JSON file format.'''
    image_name= Path(img_path).stem

    return {"version": "5.1.1", "flags": {},  "shapes":[],
              "imagePath": image_name+'.png',
              "imageData": None,
              "imageHeight": 780,
              "imageWidth": 1920 }, image_name

def write_json(save_path,file_name,data_example):
        '''
        This function writes a JSON file that contains the annotations.
            Parameters:
                save_path: A path to save the JSON files in.
                file_name: A name for the JSON file.
                data_example: A dictionary containing annotations in labelme-compatible format.
        '''
        completeName = os.path.join(save_path, file_name+".json") 
        with open(completeName, "w") as outfile:
            outfile.write(json.dumps(data_example,indent=2)) 

def hsv_extractor(img,x,y):
    '''
    This function extracts the Hue, Saturation, and Value from an HSV image at a given point (x,y).
        Parameters:
            img: A path of an image or an array representing a BGR image.
            x: The x coords of the point to be extracted.
            y: The y coords of the point to be extracted.
        return
            Array contining Hue, Saturation, and Value of the point.
    '''
    # Load the image
    if isinstance(img,np.ndarray):
        image=img
    else:
        image =cv2.imread(img)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv_image)
    # Display the HSV values at a specific pixel (x, y)
    x = int(x)  # Example x-coordinate
    y = int(y)  # Example y-coordinate

    h_value = h[y, x]
    s_value = s[y, x]
    v_value = v[y, x]
    return np.array((h_value,s_value,v_value))


def yellow_lane_marking_checker(image,x_point,y_point):
    '''
    This function takes an image and a point, then checks to see if the point, which is a bixel, has a yellow color,
    then returns the annotation of this point as yellow_lane_marking if it does, otherwise it returns white_lane_marking.
        Parameters:
            image: A path of an image or an array representing a BGR image.
            x_point: The x coords of the point to be extracted.
            y_point: The y coords of the point to be extracted.
        return
            The annotation of the polygon represented by that point.
    '''
    # Define lower and upper bounds for yellow color
    lower_yellow = np.array([0, 80, 100])
    upper_yellow = np.array([60, 255, 255])

    center_point_hsv_value= hsv_extractor(image,x_point,y_point)
    
    yellow_check=cv2.inRange(center_point_hsv_value,lower_yellow.astype(np.uint8),upper_yellow.astype(np.uint8))
    
    if int(np.sum(yellow_check))==765: # H= 255, S=255, V=255 
        return 'yellow_lane_marking'
    else: 
        return 'white_lane_marking'
    

def calculate(array):
    '''
    This function takes polygon vertices as input and computes the mean of the x and y coords,
    then subtracts this mean from the array and sums the absulot value of this subtraction. Finally,
    it determines whether this value is greater than 200.
        Parameters:
            array: Array containing the vertices of a polygon
        return:
            A boolean True if the summation greater than 200.
    '''
    mean_x = np.mean(array[:,0])
    mean_y = np.mean(array[:,1])
    mean= np.array([mean_x,mean_y])
    result = np.abs(array - mean)
    
    sumation=np.sum(result)
    result= sumation>200
    return result


def process_polygon_array(polygon):
    '''
    This function performs a simple processing on a set of arrays.
        Parameters:
            polygon: Array containing polygon vertices.
        return:
            Array containing vertices of a processed polygon.
    '''
    polygon= np.vstack(polygon).squeeze().reshape(-1,2)
    return polygon

def polygon_coordinates(polygon):
    '''
    This function takes polygon vertices and processes and refines them to produce a closed polygon.
    Depending on the number of vertices, this function fits the convex hull algo to find the vertices
    of the polygon that have all of the points lying in it (Convexity).
        Parameters:
            polygon: Array containing the vertices of a polygon.
        return
            Array containing vertices of a refined polygon.
        '''
    if polygon.shape[0]<3:
        
        # Draw the polygon
        polygon = plt.Polygon(polygon)

        #get_polygon_coordinates
        polygon=polygon.xy
        polygon=polygon.astype(np.int32)
        
        return polygon
    
    else:     
        # Compute the convex hull
        hull = ConvexHull(polygon)

        # Get the vertices of the convex hull
        hull_points = polygon[hull.vertices]

        # Draw the polygon
        polygon = plt.Polygon(hull_points)

        #get_polygon_coordinates
        polygon=polygon.xy
        polygon=polygon.astype(np.int32)

        return polygon
    

    
def count_inflection_points(second_derevetave):
    '''
    This function counts the number of inflection points.
        Parameters:
            second_derevetave: Array containing the second derevetave of spline.
        return:
            The total number of inflection points.
    '''
    # Count the sign changes in the second derevetave    
    num_inflection_points = np.count_nonzero(np.diff(np.sign(second_derevetave)) != 0)
    return num_inflection_points



def spline_second_derevetave(polygon):
    '''
    This function fits a spline through polygon vertices, then takes the line's second derevetave to get the conceve points,
    and finally counts the number of inflection points, which is the number of times the function changed its sign.
        Parameters:
            Polygon: Array containing the vertices of a polygon
        return:
            Total number of inflection points.
    '''
    try:
        polygon_points = polygon.squeeze()
        # Convert points to parametric representation
        tck, u = splprep(polygon_points.T, s=0.1, per=0,k=3)
        # Calculate the concavity of the spline using the second derevetave
        spline_slope = splev(u, tck, der=2)
        # Access the x and y components of the slope separately
        slope_x, slope_y = spline_slope[0], spline_slope[1]
        return count_inflection_points(slope_x)+count_inflection_points(slope_y)
    except (ValueError, TypeError) as e:
        return 0 