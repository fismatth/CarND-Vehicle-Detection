# Writeup

---

## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img_sliding_windows]: ./output_images/sliding_windows.jpg
[img_detections_1]: ./output_images/test1_detected_cars.jpg
[img_detections_2]: ./output_images/test2_detected_cars.jpg
[img_detections_3]: ./output_images/test3_detected_cars.jpg
[img_detections_4]: ./output_images/test4_detected_cars.jpg
[img_detections_5]: ./output_images/test5_detected_cars.jpg
[img_detections_6]: ./output_images/test6_detected_cars.jpg
[test5]: ./test_images/test5.jpg
[test5_heat]: ./output_images/test5_heatmap.jpg
[test5_thresh_heat]: ./output_images/test5_threshold_heatmap.jpg
[test5_heat_add]: ./output_images/test5_heatmap_added.jpg
[final_video]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Histogram of Oriented Gradients (HOG)

### 1. Extracting features from images

The code to extract features in the images is contained in [extract_features.py](extract_features.py). I used a combination of color features, color histogram features and HOG features.
To extract features from a single image the function single_img_features() is used. The function extract_features() computes the features for a list of filenames. 
The different parameters have been tuned by training a classifier with the extracted features (optimizing the validation accuracy).


### 2. Train classifier and optimize parameters

The code to train a classifier is contained in [classify.py](classify.py). I trained a linear SVM, using the GridSearchCV class provided by sklearn to automatically optimize the gamma and C parameter.
For the values I tried, gamma = 0.1 and C = 1 worked best. 
The classifier extracts the features of the whole data set of vehicles and non-vehicles and normalizes them using the StandardScaler class from sklearn. The data is then splitted into a training and validation set.
Then, the classifier is trained and the trained classifier together with the scaling is stored to a pickle file. 

The main issue was to find good parameters for the feature extraction. With the following parameters (see also [constants.py](constants.py)), I got a validation accuracy over 99%:

* color features:
	* color space: BGR
	* image size: 16 x 16
* color histogram features:
	* color space: BGR
	* number of bins: 64
* HOG features
	* color space: 'YCrCb'
	* orientations: 9
	* pixels per celss: 8x8
	* celss per block: 2x2
	* used channels: all


### Sliding Window Search

#### 1. Scales and overlaps

To generate a list of windows to search on, the function slide_window() in [tracking.py](tracking.py) is used. Parameters for the different scales and overlaps as well as start and stop positions are defined in [constants.py](constants.py).
There are two main points I considered to tune these parameters:
* There have to be enough sliding windows to detect cars of different sizes at different places
* There should not be more windows than needed to reduce number of false positives

I finally decided to use 4 different scales (quadratic windows with 80, 128, 170, 230 pixels) with overlaps of 0.5 except for the largest scale where I used an overlap of 0.6.
The search space in y-direction is approximately the bottom half of the image (excluding some pixels at the bottom for the smallest scale and some pixels at the top for the three larger scales).
In x-direction, I excluded 200 pixels from the left and from the right for the smallest scale and used the complete range for the three larger scales.
Here is an image showing all sliding windows:

![alt text][img_sliding_windows]

### 2. Detections on single images

Here are some example images showing the sliding windows which have been classified as cars:

![alt text][img_detections_1]
![alt text][img_detections_2]
![alt text][img_detections_3]
![alt text][img_detections_4]
![alt text][img_detections_5]
![alt text][img_detections_6]

---

### Video Implementation

#### 1. Final video output.

Here's a [link to my video result](./project_video_annotated.mp4) with estimated bounding boxes drawn around detected cars.


#### 2. Tracking and filtering

The tracking is implemented in the VehicleTracker class in [tracking.py](tracking.py). The VehicleTracker manages a list of objects of the TrackedVehicle class, which can be accepted as cars or not (yet) - depending on a score value.
For each frame, we run the sliding window search and generate a heat map using the bounding boxes classified as cars. As bounding boxes can also contain areas which are not part of the car, we update the heat map in a weighted fashion,
such that the most heat is added at the center of the bounding box (using the tensor product of 1D linear hat functions - see function add_heat_weighted() in [tracking.py](tracking.py)). 
The heat map is then normalized (especially to reduce the influence of frames with many (possibly false) detections) and added to the global heat map, which is also normalized.
Afterwards, a threshold is applied to filter out single detections (which are probably false positives). The thresholded heat map is labeled using scipy.ndimage.measurements.label() and bounding boxes around the labeled areas are computed.
The resulting bounding boxes are then used to update already tracked objects (function update() of class TrackedVehicle). 
For each tracked object, we go through the detections of the current frame and test if the overlap exceeds some threshold. If so, we update the bounding box of the tracked object with the bounding box of the current detection (averaging with a learning rate).
The score of a tracked object is incremented with each matched detection and decremented if we have no match in the current frame. A tracked object is then accepted as car if the score is at least 20. If it has not been accepted before, we additionaly check if the 
iamge extracted from the currently estimated bounding box is classified as car.
We finally remove tracked objects if their score value is too low.

---

To get an idea of how the pipeline works, here are some visualizations of the main steps:

For the following image:

![alt text][test5]

This is the heat map:

![alt text][test5_heat]

Heat map added to the image:

![alt text][test5_heat_add]

Thresholded heat map:

![alt text][test5_thresh_heat]

Video of the global heat map:

[Global heat map](./project_video_annotated_heatmap.mp4)

Video showing the currently tracked objects with their score (red boxes are not (yet) accepted due to score, green boxes have high enough score but extracted image has not been classified as car):

[Visualize filtered](./project_video_annotated_vis_filtered.mp4)

---

### Discussion

Here are some main issues of the current implementation:
* Cars are partially detected quite late due to the conservative acceptance criterias
* Bounding boxes only cover part of the tracked car or cover larger areas which don't belong to the car
* Partially hidden cars are not detected (detections are matched to car in front)
* Delayed movement of estimated bounding boxes for accelerating cars

We could try the following to improve:
* Add sliding windows locally around tracked cars to get better bounding box estimations
* Use more advanced techniques to match detections (e.g. re-use features like color histogram to verify that bounding boxes include the same car)
* Optimize classifier (tune parameters, try different features)
* With a better working pipeline overall, we should be able to use less conservative acceptance criterias (especially lower threshold for score)


