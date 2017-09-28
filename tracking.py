import cv2
import numpy as np
from scipy.ndimage.measurements import label

from constants import search_sizes, x_start_stop, y_start_stop, overlaps, bb_overlap_match
from extract_features import single_img_features

# get the center of given bounding box
def get_bounding_box_center(bb):
    x_center = 0.5 * (bb[0][0] + bb[1][0])
    y_center = 0.5 * (bb[0][1] + bb[1][1])
    return np.float32([x_center, y_center])

# get area in pixels of given bounding box
def get_area(bb):
    return (bb[1][0] - bb[0][0] + 1) * (bb[1][1] - bb[0][1] + 1)

# compute overlap of two bounding boxes
def overlap(bb1, bb2):
    x1_min = min(bb1[0][0], bb1[1][0])
    x2_min = min(bb2[0][0], bb2[1][0])
    x1_max = max(bb1[0][0], bb1[1][0])
    x2_max = max(bb2[0][0], bb2[1][0])
    y1_min = min(bb1[0][1], bb1[1][1])
    y2_min = min(bb2[0][1], bb2[1][1])
    y1_max = max(bb1[0][1], bb1[1][1])
    y2_max = max(bb2[0][1], bb2[1][1])
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return 0.0
    top_left_x = max(x1_min, x2_min)
    top_left_y = max(y1_min, y2_min)
    bottom_right_x = min(x1_max, x2_max)
    bottom_right_y = min(y1_max, y2_max)
    overlap_bb = ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
    mean_area = 0.5 * (get_area(bb1) + get_area(bb2))
    return get_area(overlap_bb) / mean_area

# class to represent a tracked object which is potentially a vehicle (will be accepted as such if score is high enough - currently 20)
class TrackedVehicle:
    def __init__(self, bb):
        # estimated bounding box of the tracked object
        self.bb = bb
        self.score = 1
        # has the object been accepted as vehicle?
        self.accepted = False
    
    # extract image of tracked object using the estimated bounding box
    def extract_car_image(self, frame):
        return frame[self.bb[0][1]:self.bb[1][1], self.bb[0][0]:self.bb[1][0]]
    
    # update tracked objects with newly detected bounding boxes
    def update(self, frame, candidates):
        found_match = False
        matched = np.array([False,] * len(candidates))
        tmp_bb = ((0.0, 0.0), (0.0, 0.0))
        num_matched = 0
        for i, c in enumerate(candidates):
            if overlap(self.bb, c) > bb_overlap_match:
                self.score = min(self.score + 1, 40)
                matched[i] = True
                found_match = True
                # update the bounding box
                tmp_bb = ((tmp_bb[0][0] + c[0][0], tmp_bb[0][1] + c[0][1]), (tmp_bb[1][0] + c[1][0], tmp_bb[1][1] + c[1][1]))
                num_matched += 1
        if num_matched > 0:
            tmp_bb = ((tmp_bb[0][0] / num_matched, tmp_bb[0][1] / num_matched), (tmp_bb[1][0] / num_matched, tmp_bb[1][1] / num_matched))
            learning_rate = 0.95
            tl_x = max(0, int(learning_rate * self.bb[0][0] + (1.0 - learning_rate) * tmp_bb[0][0]))
            tl_y = max(0, int(learning_rate * self.bb[0][1] + (1.0 - learning_rate) * tmp_bb[0][1]))
            br_x = min(frame.shape[1], int(learning_rate * self.bb[1][0] + (1.0 - learning_rate) * tmp_bb[1][0]))
            br_y = min(frame.shape[1], int(learning_rate * self.bb[1][1] + (1.0 - learning_rate) * tmp_bb[1][1]))
            self.bb = ((tl_x, tl_y), (br_x, br_y))
        if not found_match:
            self.score -= 1
        return matched
        
# class to manage and update tracked vehicles
class VehicleTracker:
    def __init__(self, clf):
        self.heatmap = None
        # classifier to detect vehicles on images
        self.clf = clf
        # create list of sliding windows where we will search on in every frame
        self.sliding_windows = []
        for i in range(len(search_sizes)):
            sw_i = slide_window(x_start_stop[i], y_start_stop[i], (search_sizes[i], search_sizes[i]), (overlaps[i], overlaps[i]))
            self.sliding_windows += sw_i
        # init the tracked objects
        self.tracked_vehicles = []
        
    # Sliding window search for given image
    def search_windows(self, img):
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in self.sliding_windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = single_img_features(test_img)
            #6) Predict using your classifier (classifier scales the features)
            prediction = self.clf(features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                #cv2.imwrite('output_images/detections/({},{})-({},{}).jpg'.format(window[0][1], window[1][1], window[0][0], window[1][0]), test_img)
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
        
    def __call__(self, img, vis_filtered_detections=False, vis_heatmap=False):
        if self.heatmap is None:
            # init initial heat map
            self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        # sliding window search
        detections = self.search_windows(img)
        # compute heat map for current frame
        heatmap_frame = np.zeros_like(self.heatmap)
        heatmap_frame = add_heat_weighted(heatmap_frame, detections)
        # scale the heat maps by their max. value but at least by 1.5
        min_scaling = 1.75
        if heatmap_frame.max() > min_scaling:
            heatmap_frame /= heatmap_frame.max()
        else:
            heatmap_frame /= min_scaling
        #return heatmap_frame * 255
        #return cv2.addWeighted(img, 1.0, np.array(cv2.merge((heatmap_frame * 255, heatmap_frame * 255, heatmap_frame * 255)), np.uint8), 1.0, 0)
        # update global heat map
        self.heatmap += heatmap_frame
        if self.heatmap.max() > min_scaling:
            self.heatmap /= self.heatmap.max()
        else:
            self.heatmap /= min_scaling
        # apply threshold to global heat map
        threshold_heatmap = apply_threshold(self.heatmap, 0.5)
        return threshold_heatmap * 255
        #return cv2.addWeighted(img, 1.0, np.array(cv2.merge((threshold_heatmap * 255, threshold_heatmap * 255, threshold_heatmap * 255)), np.uint8), 1.0, 0)
        # label heat map regions and compute bounding boxes
        labels = label(threshold_heatmap)
        candidates = get_bboxes(labels)
        # update each tracked object with the new detections
        for v in self.tracked_vehicles:
            matched = v.update(img, candidates)
            try:
                candidates = candidates[matched == False]
            except IndexError:
                candidates = np.array([])
        # add the non-matched detections to the tracked objects
        while len(candidates) > 0:
            c = candidates[0]
            self.tracked_vehicles.append(TrackedVehicle(c))
            candidates = candidates[1:]
            matched = self.tracked_vehicles[-1].update(img, candidates)
            try:
                candidates = candidates[matched == False]
            except IndexError:
                candidates = np.array([])
        # go through tracked objects and accept them as cars depending on their score
        for v in self.tracked_vehicles:
            if v.score >= 20:
                car_img = v.extract_car_image(img)
                car_img = cv2.resize(car_img, (64, 64))
                features = single_img_features(car_img)
                # if v not yet accepted, check if extracted image is classified as vehicle
                if v.accepted or self.clf(features) == 1:
                    # object accepted as vehicle
                    v.accepted = True
                    cv2.rectangle(img, v.bb[0], v.bb[1], (0,0,255), 6)
                    cv2.putText(img, 'score: {}'.format(v.score), v.bb[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
                elif vis_filtered_detections:
                    # object not accepted as vehicle
                    cv2.rectangle(img, v.bb[0], v.bb[1], (0,255,0), 6)
                    cv2.putText(img, 'score: {}'.format(v.score), v.bb[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
            elif vis_filtered_detections:
                try:
                    cv2.rectangle(img, v.bb[0], v.bb[1], (255,0,0), 6)
                    cv2.putText(img, 'score: {}'.format(v.score), v.bb[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
                except:
                    pass
        # remove objects with low score
        self.tracked_vehicles = [v for v in self.tracked_vehicles if v.score >= -2]
        if vis_heatmap:
            merged_hm = cv2.merge((self.heatmap * 255, self.heatmap * 255, self.heatmap * 255))
            heat_overlay = np.array(merged_hm, np.uint8)
            return cv2.addWeighted(img, 1, heat_overlay, 0.5, 0)
        else:
            return img

def add_heat_weighted(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # add higher weight at center of bounding boxes (linear hat fct. in x and y direction)
        center = get_bounding_box_center(box)
        w, h = box[1][0] - box[0][0] + 1, box[1][1] - box[0][1] + 1
        w_half, h_half = 0.5 * w, 0.5 * h
        for x in range(box[0][0], box[1][0]):
            for y in range(box[0][1], box[1][1]):
                heatmap[y, x] += 1.0 - (abs(center[0] - x) / w_half) * (abs(center[1] - y) / h_half)
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    threshold_heatmap = np.copy(heatmap)
    threshold_heatmap[threshold_heatmap <= threshold] = 0.0
    # Return thresholded map
    return np.clip(threshold_heatmap, 0, 255)

# compute bounding boxes for the labels
def get_bboxes(labels):
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        area = get_area(bbox)
        if area >= 16:
            bbox_list.append(bbox)
    # Return list of bounding boxes
    return np.array(bbox_list)


# Define a function that takes
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(x_start_stop, y_start_stop, xy_window, xy_overlap):
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
