import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import folium


def convert_to_lat_lon(x, y, map_center):
    """
    Converts pixel coordinates to approximate latitude and longitude
    based on the provided map center.
    """
    lat = map_center[0] + y * 0.0001  # Adjust latitude
    lon = map_center[1] + x * 0.0001  # Adjust longitude
    return [lat, lon]


# Parameters for detecting good features to track
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for the Lucas-Kanade Optical Flow algorithm
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Open the video file
cap = cv2.VideoCapture('video1.mp4')

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to load video.")
    exit()

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect initial good features to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing tracking lines
mask = np.ones_like(old_frame)

# Initialize a list to store all tracked points
tracked_points = []

# Loop through all frames of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for the detected features
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        # Select points where the status is 1 (good tracking points)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Append the new points to the tracked points list
        tracked_points.append(good_new)

        # Draw the tracking lines and points on the current frame
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Update the old frame and old points for the next iteration
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Display the tracking frame
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break
    else:
        print("Failed to compute optical flow for the frame.")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Check if any points were tracked
if len(tracked_points) > 0:
    # Combine all tracked points into a single array
    all_points = np.concatenate(tracked_points)
    x_coords = all_points[:, 0]  # Extract x-coordinates
    y_coords = all_points[:, 1]  # Extract y-coordinates

    # Perform spline interpolation to smooth the tracked path
    t = np.arange(len(x_coords))
    spl_x = splrep(t, x_coords, s=0)
    spl_y = splrep(t, y_coords, s=0)

    # Generate predicted coordinates using interpolation
    pred_t = np.linspace(0, len(x_coords) - 1, num=500)
    pred_x = splev(pred_t, spl_x)
    pred_y = splev(pred_t, spl_y)

    # Plot the tracked and predicted path
    plt.figure(figsize=(10, 6))
    plt.plot(pred_x, pred_y, label="Predicted Drone Path", color='blue')
    plt.scatter(x_coords, y_coords, c='red', label="Tracked Points", s=10)
    plt.title("Predicted Drone Path")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()

    # Save the plot to a file
    plt.savefig("drone_path_plot.png")
    plt.close()

    # Generate a map visualization
    map_center = [37.7749, -122.4194]  # Example coordinates (San Francisco)
    m = folium.Map(location=map_center, zoom_start=14)

    # Plot the predicted path as markers on the map
    for x, y in zip(pred_x, pred_y):
        lat_lon = convert_to_lat_lon(x, y, map_center)
        folium.CircleMarker(location=lat_lon, radius=5, color='blue', fill=True).add_to(m)

    # Save the map to an HTML file
    m.save("drone_path_map.html")
else:
    print("No points were tracked in the video.")
