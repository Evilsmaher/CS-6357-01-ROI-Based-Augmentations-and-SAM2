import cv2
import numpy as np
import argparse

def region_growing(image, markers, similarity_threshold=15):
    """
    Perform region growing based on pixel similarity (intensity).
    This will grow regions based on similarity and update the markers array.
    """
    updated_markers = markers.copy()
    
    for label in np.unique(markers):
        if label == 0:  # Skip the background
            continue
        
        region_mask = (markers == label).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            for point in contour:
                x, y = point[0]
                current_intensity = image[y, x]
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                        neighbor_intensity = image[ny, nx]
                        
                        if abs(int(current_intensity) - int(neighbor_intensity)) < similarity_threshold:
                            updated_markers[ny, nx] = label
    
    return updated_markers

def process_image(image_path, binary_path, output_image_path):
    image = cv2.imread(image_path)
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Morphological opening to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 1: Perform distance transform and threshold
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
    
    # Step 2: Morphological closing
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: Create background and unknown regions
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    
    # Ensure that both sure_fg and sure_bg are uint8 for subtraction
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = sure_bg.astype(np.uint8)
    
    # Step 4: Subtract to find unknown regions
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 5: Use Sobel gradient for boundary detection
    grad_x = cv2.Sobel(sure_fg, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(sure_fg, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    grad_mag = grad_mag.astype(np.uint8)

    # Step 6: Initial markers for watershed
    markers = np.zeros_like(sure_fg, dtype=np.int32)
    markers[sure_fg == 255] = 1
    markers[sure_bg == 255] = 2
    markers[unknown == 255] = 0
    
    # Step 7: Define watershed markers based on contours
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], -1, (i + 1), thickness=cv2.FILLED)

    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 8: Apply watershed
    cv2.watershed(image, markers)

    # Region growing based on similarity
    updated_markers = region_growing(gray_image, markers)

    # Hough Circle Transform to detect small fragments
    circles = cv2.HoughCircles(
        gray_image, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=20,  
        param1=50, 
        param2=30, 
        minRadius=1, 
        maxRadius=5
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            markers[y - r:y + r, x - r:x + r] = -1

    # Assign colors to segments
    output_image = image.copy()
    num_segments = np.max(updated_markers) + 1
    for label in range(1, num_segments):
        output_image[updated_markers == label] = np.random.randint(0, 256, size=3)

    output_image[updated_markers == -1] = [0, 0, 255]  # Mark watershed borders

    cv2.imwrite(output_image_path, output_image)
    print(f"Segmented image saved as {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment and save regions in an image while ignoring black pixels.")
    parser.add_argument("original_image", type=str, help="Path to the input image.")
    parser.add_argument("binary_image", type=str, help="Path to the binary mask image.")
    parser.add_argument("output_image", type=str, help="Path to the output image.")
    args = parser.parse_args()

    process_image(args.original_image, args.binary_image, args.output_image)
