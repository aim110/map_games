import cv2
import numpy as np
import gradio as gr

def detect_points_of_interest(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply adaptive thresholding to detect high contrast regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        # Get the minimum enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        # Calculate contour area and circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

        # Filter based on radius and circularity
        if 10 < radius < 100 and circularity > 0.7:
            # Draw the detected POI
            cv2.circle(image, (int(x), int(y)), radius, (255, 0, 0), 2)
            points.append((int(x), int(y), radius))

    return image, points

def find_non_colliding_circle(image_shape, points, radius, min_distance):
    height, width = image_shape[:2]

    # Check grid of potential positions for the new circle
    for y in range(radius, height - radius, 5):  # Step by 5 pixels for efficiency
        for x in range(radius, width - radius, 5):
            # Check if this circle is at least 'min_distance' away from all existing POIs
            if all(np.hypot(px - x, py - y) > (pr + radius + min_distance) for px, py, pr in points):
                return (x, y)

    return None

def process_image(image, radius, min_distance):
    # Detect points of interest and draw blue circles around them
    image_with_poi, points = detect_points_of_interest(image)

    # Try to find a place for the red circle that doesn't collide with any POIs
    position = find_non_colliding_circle(image.shape, points, radius, min_distance)

    if position:
        # Draw the red circle at the found position
        cv2.circle(image_with_poi, position, radius, (0, 0, 255), 2)
        result_text = f"Red circle placed at {position}."
    else:
        result_text = "Can't place the red circle without collision."

    return image_with_poi, result_text

def gradio_app(image, radius, min_distance):
    # Process the uploaded image with the specified radius and minimum distance
    return process_image(image, radius, min_distance)

# Gradio interface
interface = gr.Interface(
    fn=gradio_app,
    inputs=[
        gr.Image(type="numpy", label="Upload Map Image"),
        gr.Slider(minimum=100, maximum=1000, step=1, value=500, label="Circle Radius (px)"),
        gr.Slider(minimum=0, maximum=100, step=1, value=30, label="Minimum Distance from POIs (px)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Output Image"),
        gr.Textbox(label="Result")
    ],
    title="Map POI Detection and Circle Placement",
    description="Upload a map image and choose a circle radius and minimum distance from detected POIs. The app detects points of interest (POI) and places a non-colliding circle."
)

if __name__ == "__main__":
    interface.launch()

