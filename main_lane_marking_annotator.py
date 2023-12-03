import argparse
import numpy as np 
import cv2 
import torch
from tqdm import tqdm
import warnings
from src.general_utiles import *
from src.helper_finding_marking_centers import *
from src.helper_polygons_processing_filtering import *

warnings.simplefilter("ignore")

def lane_marking_annotator(source_directory_path, save_path,sam_checkpoint):
    '''
    This is the primary function for annotating the lane markings presented in the images. 
    The annotations will be written in a JSON file, which will include the polygon vertices as well as the class of the
    polygon, and info about the annotated image. The resulting JSON file will be labelme-compatible.
        Parameters:
            source_directory_path: A path to the source directory containing images, str.
            save_path: A path of the directory in which the JSON files containing annotations will be saved, str.
            sam_checkpoint: A path to sam checkpoint (SAM model trained weights), str.
    '''
    torch.cuda.empty_cache()
    print('Starting Process')
    predictor= model_loader_predictor(sam_checkpoint=sam_checkpoint)
    print('Model is loaded')
    all_images=images_paths(images_directory=source_directory_path)
    print(f'Total number of images: {len(all_images)}')


    for num, img_id in tqdm(enumerate(range(len(all_images))), total=len(all_images), desc="Processing"):
        
        try:
            try:
                centers=find_cluster_centers(image_path=all_images[img_id])


            except (ValueError,IndexError) as e:



                img,contours,total_time=\
                contour_drawer(all_images[img_id],control=True, upper=20, lower=1,epsilon=0.01,hieght_limit=150
                                                    ,number_of_inflection_points=12)

                centers= centers_calculator(contours=contours)

                if centers =='continue':
                    continue

                #filter out the centers based on the limits 
                limits= img_limits(centers=centers)
                centers=np.vstack([np.mean(centers[limit],axis=0) for limit in limits if True in limit])

                if len(centers)==0:
                    continue

                centers=np.vstack(centers)

            torch.cuda.empty_cache()

            image_org = cv2.imread(all_images[img_id])
            image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

            predictor.set_image(image)

            input_point = centers
            input_label = np.ones_like(centers[:,1])

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            approx=image_mask_contours(masks=masks)
            #incase there is no contours skip this image
            if len(approx)==0:
                continue

            #filtering out the data with mean sum less than 50, the idea behind this function is to remove any array
            #that does not have a high variance, which means the array is representing a point most likely
            indx=[calculate(i.reshape(-1,2)) for i in approx]

            #set first limit to be 850 on the x, second limit from 850-1250, last limit from 1250 all the way to the end
            #the polygons will be drawn for each part seperatly
            #filtering data based on partitioning the image

            filtered_polygons=np.array(approx)[indx]

            filtered_objects=filter_polygon_objects(filtered_polygons=filtered_polygons,approx=approx)

            #get the polygons coordinates
            polygons=[polygon_coordinates(process_polygon_array(poly)) for poly in filtered_objects if len(poly)>0]

            #filter the polygons based on the area
            polygons=[polygon for polygon in polygons if (polygon_area(polygon) <100000) & (polygon_area(polygon) >500)]

            if not len(polygons)==0:

                torch.cuda.empty_cache()
                data_example, image_name= json_file_data_format(img_path=all_images[img_id])       

                for obj in polygons:
                    #check the color of the center point of the polygon to assign the right class
                    center_point=np.mean(obj,axis=0)
                    label=yellow_lane_marking_checker(image=image_org,x_point=center_point[0],y_point=center_point[1])

                    #fill the annotation data
                    data_example['shapes'].append({'label':label,
                    'points':obj.tolist(), 'group_id': None, 
                    'shape_type':"polygon",
                    "flags": {}})

                write_json(save_path=save_path,file_name=image_name,data_example=data_example)

                print(all_images[img_id], f'This is image number {num}')

            else:
                continue
            

        except Exception as e:
            continue            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
    '''
    Lane Marking Annotator:
    This is the primary function for annotating the lane markings presented in the images. 
    The annotations will be written in a JSON file, which will include the polygon vertices as well as the class of the 
    polygon, and info about the annotated image. The resulting JSON file will be labelme-compatible.
        Parameters:
            source_directory_path: A path to the source directory containing images, str.
            save_path: A path of the directory in which the JSON files containing annotations will be saved, str.
            sam_checkpoint: A path to sam checkpoint (SAM model trained weights), str.
    ''' )

    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    args = parser.parse_args()
    config = parse_config(args.config)

    lane_marking_annotator(config['source_directory_path'], config['save_path'], config['sam_checkpoint'])