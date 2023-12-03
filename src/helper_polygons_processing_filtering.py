import numpy as np 
import cv2 
from .general_utiles import spline_second_derevetave

def polygon_area(polygon):
    '''This function finds the area of a polygon.'''
    x = polygon[:, 0]
    y = polygon[:, 1]
    area = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    return abs(area) / 2

    
def mask_area(mask):
    '''
    This function computes the area of a mask within a binary image.
        Parameters:
            mask: Array representing a binary image.
        return:
            The area percent of the non zero bixels with respect to the binary image. 
    '''
    total_area=1497600 #1920*780 image width*image height
    mask_area=np.count_nonzero(mask)
    mask_area_percent= mask_area/ total_area
    return mask_area_percent

def mask_filter(masks):
    '''
        This function removes masks based on their area percentage.
        Parameters:
            masks: Array representing a binary images.
        return:
            The filtered masks that satisfies the criteria.
        '''
    masks_areas=np.array([mask_area(mask) for mask in masks[:-1]])

    if False in masks_areas<.04:
        choosen_mask=np.argmax(masks_areas)
        choosen_mask=masks[choosen_mask]
    else:
        choosen_mask=np.argmin(masks_areas)
        choosen_mask=masks[choosen_mask]
        
    return choosen_mask



def image_mask_contours(masks):
    '''
    This function takes a binary image that contains a polygon mask, finds the contours of that mask,
    and then uses those contours to approximate the polygons vertices. After approximating the polygons vertices,
    it will filter out polygons that have a peak to peak differance less than 100 on the y axis,
    as well as polygons that has a peak to peak difference on the x axis more than 450 or less than 35.
    Furthermore, the function filters any polygon that has more than 20 inflection points.
        Parameters:
            masks: A list containing binary images, these binary images contain masks of polygons.
        return
            Array of the approximated polygon vertices.

    '''
    image=np.where(mask_filter(masks=masks)==False,0,255)
    image=image.astype(np.uint8).squeeze()

    # Apply edge detection or contour detection algorithm on the extracted image
    edges = cv2.Canny(image, 50, 140)
    # Find contours of the mask
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approx = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in contours]#0.01 for epsilon


    approx=[contour for contour in approx if (np.ptp(contour[:,0,1])>100) and ((np.ptp(contour[:,0,0])<450) or (np.ptp(contour[:,0,0])>35))]#50
    approx=[contour for contour in approx if spline_second_derevetave(contour) < 20]

    return approx

def filter_polygon_objects(filtered_polygons,approx):
    '''This function will filter a set of polygons by partitioning the image based on its width.'''
    top_right_corner=[poly for poly in filtered_polygons if (cv2.boundingRect(poly)[0]>1500) & (cv2.boundingRect(poly)[1]<100)]
    if len(top_right_corner):
        indices_of_corner=np.nonzero([np.array_equal(corner_arr, arr) for corner_arr in top_right_corner for arr in approx])[0]
        indices_of_corner=indices_of_corner-np.arange(indices_of_corner.size)*len(approx)
        try:
            filtered_polygons=np.delete(filtered_polygons,indices_of_corner)

        except IndexError as e:
            filtered_polygons

    if len(filtered_polygons):
        if filtered_polygons[0].ndim:
            filtered_lines_part_1=np.array([lines for lines in filtered_polygons if cv2.boundingRect(lines.squeeze())[0]<850])
            filtered_lines_part_2=np.array([lines for lines in filtered_polygons \
                                   if (cv2.boundingRect(lines.squeeze())[0] <1250) &(cv2.boundingRect(lines.squeeze())[0] >850)])
            filtered_lines_part_3=np.array([lines for lines in filtered_polygons if cv2.boundingRect(lines.squeeze())[0] >1250])

            filtered_objects=[filtered_lines_part_1,filtered_lines_part_2,filtered_lines_part_3, top_right_corner]

        else:
            filtered_objects=top_right_corner
    else:

        filtered_objects=top_right_corner


    return filtered_objects