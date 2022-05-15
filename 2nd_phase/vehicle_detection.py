from helper_fun import *

my_parser = argparse.ArgumentParser()

my_parser.add_argument('path', type=str)
my_parser.add_argument('mode', type=str)

cars = []
notcars = []

# Divide up into cars and notcars
images = glob.glob('data/*/*/*.png')

for image in images:
    if 'vehicles' in image:
        cars.append(image)
    elif 'not-cars' in image:
        notcars.append(image)

print ( "Number of images containing cars:", len(cars))
print ( "Number of images not containing cars:", len(notcars))

# Define HOG parameters
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

# Spatial size and histogram parameters
spatial_size=(16, 16)
hist_bins=16

# Pick for two samples
images = [cars[0], notcars[0]]

for img_p in images:
    

    print ("Analyzing pictures: ", img_p)

    # Read in the image
    image = mpimg.imread(img_p)
    # Convert in YCrCb
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Call our function with vis=True to see an image output
    hog_features = []
    hog_images = []
    for channel in range(image_YCrCb.shape[2]):
        features, hog_image = get_hog_features(image_YCrCb[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=True, feature_vec=True)
        hog_features.append(features)
        hog_images.append(hog_image)


    # Plot the examples
    fig = plt.figure(figsize = (8,8))
    plt.subplot(221)
    plt.imshow(image)
    plt.title('Example Image')
    plt.subplot(222)
    plt.imshow(image_YCrCb[:,:,0], cmap='gray')
    plt.title('Component Y')
    plt.subplot(223)
    plt.imshow(image_YCrCb[:,:,1], cmap='gray')
    plt.title('Component Cr')
    plt.subplot(224)
    plt.imshow(image_YCrCb[:,:,2], cmap='gray')
    plt.title('Component Cb')

    # Plot the examples
    fig = plt.figure(figsize = (8,8))
    plt.subplot(221)
    plt.imshow(image)
    plt.title('Example Image')
    plt.subplot(222)
    plt.imshow(hog_images[0], cmap='gray')
    plt.title('HOG Visualization channel 0')
    plt.subplot(223)
    plt.imshow(hog_images[1], cmap='gray')
    plt.title('HOG Visualization channel 1')
    plt.subplot(224)
    plt.imshow(hog_images[2], cmap='gray')
    plt.title('HOG Visualization channel 2')

#Extract Feature and Build Classifier
print ('Extracting car features')
car_features = extract_features(cars, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
print ('Extracting not-car features')
notcar_features = extract_features(notcars, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',spatial_size, 'spatial_size' , hist_bins, 'hist_bins')
print('HOG: Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
prediction = svc.predict(X_test[0].reshape(1, -1))
t2 = time.time()
print(t2-t, 'Seconds to predict with SVM')

# Visualize a confusion matrix of the predictions
pred = svc.predict(X_test)
cm = pd.DataFrame(confusion_matrix(pred, y_test))
cm

#Save data
# Save a dictionary into a pickle file.
classifier_info = { "svc": svc, "scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell,
"cell_per_block": cell_per_block, "spatial_size": spatial_size, 'hist_bins': hist_bins }

pickle.dump( classifier_info, open( "classifier_info.p", "wb" ) )

#Import classifier model and feature extraction settings
dist_pickle = pickle.load( open("classifier_info.p", "rb" ) )
svc_l = dist_pickle["svc"]
X_scaler_l = dist_pickle["scaler"]
orient_l = dist_pickle["orient"]
pix_per_cell_l = dist_pickle["pix_per_cell"]
cell_per_block_l = dist_pickle["cell_per_block"]
spatial_size_l = dist_pickle["spatial_size"]
hist_bins_l = dist_pickle["hist_bins"]

args = my_parser.parse_args()
global mode 
global path
mode = args.mode
path = args.path
if args.mode == 'debug':
    if "mp4" in path:
        #Test on a video
        heatmaps = collections.deque(maxlen=29)
        def process_image(img):
            global heatmaps

            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            
            ystarts = [400,350,350]
            ystops = [656,570,570]
    
            # Look for cars at different scales
            scales = [1., 1.5, 2.0]

            for scale, ystart, ystop  in zip(scales,ystarts,ystops):
                box_list,out_img,out_img_windows  = find_cars(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l, pix_per_cell_l, cell_per_block_l, spatial_size_l, hist_bins_l)
                heat = add_heat(heat,box_list)

            # Append heatmap and compute the sum of the last n ones
            heatmaps.append(heat)
            sum_heatmap = np.array(heatmaps).sum(axis=0)
            # Apply the threshold to remove false positives
            heat = apply_threshold(sum_heatmap, min(len(heatmaps) * 1, 28))

            # Visualize the heatmap when displaying    
            heatmap = np.clip(heat, 0, 255)
            
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img = draw_labeled_bboxes(np.copy(img), labels)
            return draw_img

        video_output = 'output_videos/result.mp4'
        clip2 = VideoFileClip(path)
        video_clip = clip2.fl_image(process_image)
        video_clip.write_videofile(video_output, audio=False)
    else:
        #Test on Images
        #Read an image to test
        img = mpimg.imread(path)
        # Create the heat map
        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Define ROI of the images where to use the sliding windows
        ystart = 400
        ystop = 656

        # Look for cars at different scales
        scales = [1., 1.5, 2.0]

        for scale in scales:
            box_list,out_img,out_img_windows  = find_cars(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l, pix_per_cell_l, cell_per_block_l, spatial_size_l, hist_bins_l)
            heat = add_heat(heat,box_list)
            
            fig = plt.figure(figsize = (16,16))
            plt.subplot(121)
            plt.imshow(out_img)
            caption = 'Scale: ' +  str(scale)
            plt.title(caption)
            plt.subplot(122)
            plt.imshow(heat, cmap='hot')
            plt.title('Heat Map')

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        fig = plt.figure(figsize = (16,16))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

else:
    if "mp4" in path:
        #Test on a video
        heatmaps = collections.deque(maxlen=29)
        def process_image(img):
            global heatmaps

            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            
            ystarts = [400,350,350]
            ystops = [656,570,570]
    
            # Look for cars at different scales
            scales = [1., 1.5, 2.0]

            for scale, ystart, ystop  in zip(scales,ystarts,ystops):
                box_list,out_img,out_img_windows  = find_cars(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l, pix_per_cell_l, cell_per_block_l, spatial_size_l, hist_bins_l)
                heat = add_heat(heat,box_list)

            # Append heatmap and compute the sum of the last n ones
            heatmaps.append(heat)
            sum_heatmap = np.array(heatmaps).sum(axis=0)
            # Apply the threshold to remove false positives
            heat = apply_threshold(sum_heatmap, min(len(heatmaps) * 1, 28))

            # Visualize the heatmap when displaying    
            heatmap = np.clip(heat, 0, 255)
            
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img = draw_labeled_bboxes(np.copy(img), labels)
            return draw_img

        video_output = 'output_videos/result.mp4'
        clip2 = VideoFileClip(path)
        video_clip = clip2.fl_image(process_image)
        video_clip.write_videofile(video_output, audio=False)
    else:
        #Test on Images
        #Read an image to test
        img = mpimg.imread(path)
        # Create the heat map
        heat = np.zeros_like(img[:,:,0]).astype(np.float)

        # Define ROI of the images where to use the sliding windows
        ystart = 400
        ystop = 656

        # Look for cars at different scales
        scales = [1., 1.5, 2.0]

        for scale in scales:
            box_list,out_img,out_img_windows  = find_cars(img, ystart, ystop, scale, svc_l, X_scaler_l, orient_l, pix_per_cell_l, cell_per_block_l, spatial_size_l, hist_bins_l)
            heat = add_heat(heat,box_list)
            
            fig = plt.figure(figsize = (16,16))
            plt.subplot(121)
            plt.imshow(out_img)
            caption = 'Scale: ' +  str(scale)
            plt.title(caption)
            plt.subplot(122)
            plt.imshow(heat, cmap='hot')
            plt.title('Heat Map')

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        fig = plt.figure(figsize = (16,16))
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

