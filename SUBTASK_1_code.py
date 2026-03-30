import numpy as np
import cv2

def lucas_kanade_scratch(prev_img, next_img, prev_pts, window_size=15):
    """
    Manual implementation of the Lucas-Kanade Sparse Optical Flow algorithm.
    Assumes that the flow is constant within a small local window.
    """
    w = window_size // 2
    
    # --- 1. PRE-COMPUTING GRADIENTS ---
    # Ix: Change in intensity horizontally (Sobel derivative)
    # Iy: Change in intensity vertically (Sobel derivative)
    Ix = cv2.Sobel(prev_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_img, cv2.CV_64F, 0, 1, ksize=3)
    # It: Temporal gradient (How much pixel values changed between Frame A and Frame B)
    It = next_img.astype(np.float64) - prev_img.astype(np.float64)

    new_pts = []
    status = []

    # Iterate through every feature point we want to track
    for pt in prev_pts:
        x, y = pt.ravel()
        x, y = int(x), int(y)

        # --- 2. WINDOW EXTRACTION ---
        # Get gradients for a window_size x window_size area around the point
        ix = Ix[y-w:y+w+1, x-w:x+w+1].flatten()
        iy = Iy[y-w:y+w+1, x-w:x+w+1].flatten()
        it = It[y-w:y+w+1, x-w:x+w+1].flatten()

        # Handle boundary cases where the window goes outside image dimensions
        if len(ix) < (window_size**2):
            new_pts.append([x, y])
            status.append(0) # Mark as lost
            continue

        # --- 3. SOLVING THE LEAST SQUARES PROBLEM ---
        # The equation is: Ix*u + Iy*v = -It
        # We represent this as A * d = b, where d = [u, v]^T
        A = np.vstack((ix, iy)).T
        b = -it.reshape(-1, 1)

        # Compute A^T * A (a 2x2 matrix)
        ATA = A.T @ A
        
        # Check if ATA is invertible (avoids errors in flat areas like blank walls)
        if np.linalg.det(ATA) < 1e-6:
            new_pts.append([x, y])
            status.append(0)
        else:
            # Solve for displacement vector d: (A^T * A)^-1 * A^T * b
            d = np.linalg.inv(ATA) @ (A.T @ b)
            u, v = d.ravel()
            # Update the point's position based on calculated velocity
            new_pts.append([x + u, y + v])
            status.append(1) # Mark as successfully tracked

    return np.array(new_pts, dtype=np.float32), np.array(status)

# --- INITIALIZATION ---
cap = cv2.VideoCapture('OPTICAL_FLOW.mp4')

# Parameters for Shi-Tomasi Corner Detection (choosing points to track)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Read initial frame and find the first set of features
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a blank black 'mask' to draw the persistent motion lines (light beams)
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Convert current frame to grayscale for processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- TRACKING ---
    # Call our custom Lucas-Kanade function
    p1, status = lucas_kanade_scratch(old_gray, frame_gray, p0)

    # Only process points that were successfully tracked (status == 1)
    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        # --- VISUALIZATION ---
        # Draw the motion paths (the "beams")
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel() # Current position
            c, d = old.ravel() # Previous position
            
            # Draw a red line on the persistent mask connecting old and new positions
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            # Draw a green circle on the current frame at the tracked point
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

    # --- BLENDING & DISPLAY ---
    # Overlay the red line mask onto the original video frame
    img = cv2.add(frame, mask)
    cv2.imshow('Lucas-Kanade Flow (Scratch)', img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # --- PREPARE FOR NEXT FRAME ---
    # Current grayscale frame becomes the 'previous' frame for the next iteration
    old_gray = frame_gray.copy()
    # Update points: the 'new' points become the 'old' points to track from next
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
