# Unsupervised Lane Marking Annotator

This project provides a script to unsupervisedly annotate pavement lanemarking in RGB images. It uses the raw image and generates corresponding segmentation masks. The output of the script will be JSON files containing annotations and they will be saved at the specified path.

<p float="left" align="center">
  <img src="./script_output_example/algo_flowchart.png" width="550" /> 
</p>

## Installation

Follow these steps to install and run the script:

1. **Clone the Repository Locally**: Use the following command to clone the repo:
   ```shell
    git clone --recursive https://github.com/Younis-Matouq/Lane_Marking_Annotator.git
    ```

2. **Navigate to the Project Directory**: Install the required Python packages using `pip`:

    ```shell
    cd Lane_Marking_Annotator
    ```

   ```shell
    pip install -r requirements.txt
    ```
3. **Navigate to Segment Anything Directory**: Use the `cd` command to navigate into the `segment-anything` directory:

    ```shell
    cd segment-anything
    ```

4. **Install the Requirements**: Install the required Python packages using `pip`:

    ```shell
    pip install -e .
    ```

## Configuration

The script uses a `config.yaml` file to pass arguments to the main script. To run the script properly, you must update this file with your specific settings. Here's a description of what each setting does:

- `source_directory_path`: A path to the source directory containing images, str.
- `save_path`: The directory where the script will save the output JSON files.
- `sam_checkpoint`: The path to the SAM model checkpoint file. The checkpoint is available at https://github.com/facebookresearch/segment-anything

Below is an example of how your `config.yaml` file might look:

```yaml
source_directory_path: img_directory\imgs
save_path: saving_path\results
sam_checkpoint: sam_checkpoint_path\sam_vit_h_4b8939.pth
```

## Usage 
**To run the script navegate to the Lane_Marking_Annotator directory then use the following command.**

```python
python main_lane_marking_annotator.py --config config.yaml
```
    

## Output

Once you run the script, it will process the images in the specified directories,and will generate the segmentation annotations. 

The output will be a set of JSON files containing the segmentation annotations. These files will be in a format compatible with labelme.

The JSON files will be saved in the directory specified by the `save_path` parameter in the `config.yaml` file.

> :information_source: **Note:** The purpose of this algorithm is to facilitate the annotation process. It is important to note that the annotations produced by this algorithm require further evaluation for accuracy. While the algorithm may not annotate all objects, it serves as an effective starting point for creating a dataset. This dataset can then be used to train an instance segmentation model specifically for segmenting lane line markings.

## License

This project is licensed under the terms of the MIT License.


