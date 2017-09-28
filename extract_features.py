import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from constants import default_spatial_size,\
    default_hist_bins, default_hist_range, default_orient, default_pix_per_cell,\
    default_cell_per_block, default_hog_channel, default_color_space_hog,\
    default_color_space_spatial, default_color_space_hist

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def convert_color(img, color_space):
    if color_space != 'BGR':
        if color_space == 'RGB':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(img)   
    return feature_image

def get_color_features(img, color_space, size):
    # Convert image to new color space (if specified)
    feature_image = convert_color(img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

def single_img_features(img, color_space_spatial=default_color_space_spatial, color_space_hist=default_color_space_hist,
                        color_space_hog=default_color_space_hog, spatial_size=default_spatial_size,
                        hist_bins=default_hist_bins, hist_range=default_hist_range, orient=default_orient, 
                        pix_per_cell=default_pix_per_cell, cell_per_block=default_cell_per_block, hog_channel=default_hog_channel):
    # get color features
    color_features = get_color_features(img, color_space_spatial, size=spatial_size)
    feature_image = convert_color(img, color_space_hist)
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    
    feature_image = convert_color(img, color_space_hog)
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    # Append the new feature vector to the features list
    #features.append(np.concatenate((color_features, hist_features, hog_features)))
    return np.concatenate((color_features, hist_features, hog_features))
    #return hog_features
    
# Define a function to extract features from a list of images
# Have this function call get_color_features() and color_hist()
def extract_features(imgs, color_space_spatial=default_color_space_spatial, color_space_hist=default_color_space_hist,
                    color_space_hog=default_color_space_hog, spatial_size=default_spatial_size,
                    hist_bins=default_hist_bins, hist_range=default_hist_range, orient=default_orient, 
                    pix_per_cell=default_pix_per_cell, cell_per_block=default_cell_per_block, hog_channel=default_hog_channel):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        img = cv2.imread(file)
        features.append(single_img_features(img, color_space_spatial, color_space_hist, color_space_hog, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel))
        # Append the new feature vector to the features list
        #features.append(hog_features)
    # Return list of feature vectors
    return features

def merge_features(feat1, feat2):
    return np.vstack((feat1, feat2)).astype(np.float64)

def plot3d(pixels, colors_rgb,
    axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

# Define a function to compute color histogram features  
def color_hist(img, nbins, bins_range, vis = False):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if vis:
        return rhist, ghist, bhist, bin_centers, hist_features
    else:
        return hist_features


if __name__ == '__main__':
    image = cv2.imread('test_images/test1.jpg')
    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256), vis=True)

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')
        
    
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(image.shape[0], image.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(image, (np.int(image.shape[1] / scale), np.int(image.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    
    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    
    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()
    
    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()