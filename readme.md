## Writeup/Readme
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Feature Extraction

#### 1. Extracting HOG, Spatial, and Color features from the training images.

The code for this step is contained in the code cell of the IPython notebook titled "Hog, Bin, Color feature functions", methods `get_hog_features` `bin_spatial` `color_hist`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes and the resulting HOG features:

# Example HOG features for a car vs notcar

![png](/output_images/output_6_1.png)


# Visualizing the Spatial and Color features
Here are the histograms for the spatial and color features that are extracted from an image.
Spatial size 32,32. Histogram bins 32.
The features are extracted then normalized.

![png](/output_images/output_10_0.png)


# Extract all features and training the SVM
The code contained under section titled "Extract all features (Hog,Bin,Color)" and "Extract the features and train the SVM" takes the set of training images, both car and noncar, and extracts features from both. The features are stacked into one list and a standard scaling is applied.
The training sets are then randomized and split to produce randomized training and testing sets. These sets are used to train the linear SVC, resulting in an accuracy of 98.9.
The variables tested were obtained from the chapter material and were kept since they produced an acceptable accuracy. The difference here being the YUV colorspace that is used which has proven to be more precise in identifying the colors under different conditions.

    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins

    Using: 9 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 17676

    38.24 Seconds to train SVC...
    Test Accuracy of SVC =  0.989

# Sliding window search
Sliding window search was performed with the HOG subsampling method. The window search relies on the set of 3 features previously discussed.
Given a start and stop Y axis position, the function slides a box across the image in search of matching features that the SVC classifies as a 'car'. If found, a set of coordinates is generated to box that area and the function will run and retrieve create multiple rectangle coordinates until it reaches the end of the specified search range.
The code is contained in method `find_cars` which is under the section "Perform sliding window with HOG subsampling..."
Here is an example of the rectangles returned by the method, drawn over the image by the `draw_boxes` method.
    
ystart = 400
ystop = 656
scale = 1.5

![png](/output_images/output_19_1.png)


# Heatmap thresholding
As you can see, the sliding window search results in many boxes drawn over the image and sometimes may draw a box over an incorrect location. To better draw the boxes I used a heatmap to determine the most active areas in a image and used a threshold to get rid of false positives. The heatmap is generated and a threshold applied, resulting in one big 'bright' area where a box will be drawn to cover the detected vehicle, which helps clean up the final image and fix any bad detections.
The code for the heatmap threshold starts under section titled "Add heat function".
Below are example images running through the heatmap threshold process:

Initial heatmap generated:

![png](/output_images/output_22_1.png)

Heatmap with threshold 1.0:

![png](/output_images/output_25_1.png)

Final heatmap result:

![png](/output_images/output_27_1.png)

Drawing the new boxes:

![png](/output_images/output_29_1.png)

# Full pipeline to detect cars and draw rectangles
The set of functions is put into one pipeline under the method `detect_pipeline`. In this method, the pipeline is run multiple times with different scales and Y positions for the sliding window search. This helps the detection by looking for smaller boxes the further away a car might be in the image and contains some overlap to increase the accuracy.

    colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    
    ystart = 400
    ystop = 464
    scale = 1.0

    ystart = 416
    ystop = 480
    scale = 1.0

    ystart = 400
    ystop = 496
    scale = 1.5

    ystart = 432
    ystop = 528
    scale = 1.5

    ystart = 400
    ystop = 528
    scale = 2.0

    ystart = 432
    ystop = 560
    scale = 2.0

    ystart = 400
    ystop = 596
    scale = 3.5

    ystart = 464
    ystop = 660
    scale = 3.5

# Running the pipeline on the example images


![png](/output_images/output_34_0.png)


The pipeline is also run on the example project videos, the resulting videos are located under the folder output_vids.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline performs well in the given video, detecting both close and far away vehicles that are in the range. It does however take some time, longer than optimal, to identify the cars when the first appear on screen. To increase the speed and accuracy the features need to be tuned more. The size of the HOG features and the spatial/color could be improved to decrease the size of the feature vector length, speeding up the process. Other changes like using a different training dataset or a different combination of colorspace,cells per block, pixels per cell, and orientations could result in a more accurate pipeline.
