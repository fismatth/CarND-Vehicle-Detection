
# define constants to generate windows that will be used to search for cars in images 

# overlap of sliding windows
overlaps = [0.5, 0.5, 0.5, 0.6]
# sliding window sizes
search_sizes = [80, 128, 170, 230]
# start and stop positions in x-direction for sliding windows
x_start_stop = [(200, 1080), (0, 1280), (0, 1280), (0, 1280)]
# start and stop positions in y-direction for sliding windows
y_start_stop = [(360, 616), (400, 720), (400, 720), (400, 720)]


# constants for feature extraction
default_color_space_spatial='BGR'
default_color_space_hist='BGR'
default_color_space_hog= 'YCrCb'
default_spatial_size=(16, 16)
default_hist_bins=64
default_hist_range=(0, 256)
default_orient=9
default_pix_per_cell=8
default_cell_per_block=2
default_hog_channel='ALL'

# min. relative bounding box overlap to match car detections
bb_overlap_match = 0.5