from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import glob
from camera_calibration import Undistort
from classify import Classifier
from tracking import  VehicleTracker
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

undistorter = Undistort()
clf = Classifier()
tracker = VehicleTracker(clf)

def draw_boxes(img, bboxes, color=(255, 0, 0), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for bb in bboxes:
        cv2.rectangle(draw_img, bb[0], bb[1], color, thick)
    return draw_img

def convert_binary(binary):
    return np.array(cv2.merge((binary, binary, binary)),np.uint8) * 255

def process_image(img):
    # undistor image like in Advance-Lane-Lines project
    undistorted = undistorter(img)
    # update the tracked objects
    return tracker(undistorted, True, True)


if __name__ == '__main__':
    #ffmpeg_extract_subclip("project_video.mp4", 0, 8, targetname="test_video_4.mp4")
    video = False
    if video:   
        video_fname = 'project_video.mp4'
        tracker = VehicleTracker(clf)
        clip1 = VideoFileClip(video_fname)
        white_clip = clip1.fl_image(process_image)
        # since threshold 0.4 only normalize if max > 1.0
        white_clip.write_videofile(video_fname.replace('.mp4', '_annotated_vis_filtered.mp4'), audio=False)
    else:
        for fname in glob.glob('test_images/test*.jpg'):
            tracker = VehicleTracker(clf)
            img = cv2.imread(fname)
            sliding_windows_img = draw_boxes(img, tracker.sliding_windows)
            cv2.imwrite('output_images/sliding_windows.jpg', sliding_windows_img)
            result_img = process_image(img)
            #result_img = draw_boxes(img, detections)
            cv2.imwrite(fname.replace('test_images', 'output_images').replace('.jpg', '_threshold_heatmap.jpg'), result_img)
        