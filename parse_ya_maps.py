import cv2
import numpy as np
import argparse
import easyocr
import gradio as gr

# Function to process the image and find points of interest and labels
def process_image(image_path, closeness_threshold, label_distance_threshold):
    print("Loading image from:", image_path)
    # Load the image
    image = cv2.imread(image_path)

    # Global parameters
    distance_threshold = label_distance_threshold

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale.")

    # Apply GaussianBlur to reduce noise and improve object detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    print("Applied Gaussian Blur.")

    # Use adaptive thresholding to create a binary image with high contrast
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    print("Adaptive thresholding applied.")

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # List to store detected round points of interest
    points_of_interest = []

    # Iterate over contours to find round objects based on circularity and contrast
    for contour in contours:
        # Approximate the contour to check if it's roughly circular
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Calculate the area and circularity of the contour
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity > 0.7 and len(approx) > 5:  # Filter based on circularity
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                if radius > 10 and radius < 100:
                    # Draw the circle in the output image (for visualization purposes)
                    cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 4)
                    points_of_interest.append((int(x), int(y), int(radius)))
    print(f"Detected {len(points_of_interest)} points of interest.")

    # Detect all labels in the image using EasyOCR
    reader = easyocr.Reader(['en', 'ru'])
    results = reader.readtext(image)
    print(f"EasyOCR detected {len(results)} text regions.")

    # List to store combined labels
    combined_labels = []

    # Combine labels that belong to the same line
    current_line_text = ""
    current_line_coords = None

    for (bbox, text, _) in results:
        (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
        text = text.strip()
        if text:
            if current_line_coords is None:
                current_line_coords = (x, y, w, h)
                current_line_text = text
            else:
                # If the current label is on the same line (based on y-coordinate and x-coordinate proximity)
                if abs(y - current_line_coords[1]) <= distance_threshold and abs(x - (current_line_coords[0] + current_line_coords[2])) <= distance_threshold:
                    current_line_text += f" {text}"
                    current_line_coords = (
                        min(current_line_coords[0], x),
                        min(current_line_coords[1], y),
                        max(current_line_coords[0] + current_line_coords[2], x + w) - min(current_line_coords[0], x),
                        max(current_line_coords[1] + current_line_coords[3], y + h) - min(current_line_coords[1], y)
                    )
                else:
                    combined_labels.append((current_line_text, current_line_coords))
                    current_line_coords = (x, y, w, h)
                    current_line_text = text

    # Append the last line
    if current_line_text:
        combined_labels.append((current_line_text, current_line_coords))
    print(f"Combined labels into {len(combined_labels)} entries.")

    # List to store point of interest labels
    poi_labels = []

    # Iterate over combined labels and match them with points of interest
    for label_text, (x, y, w, h) in combined_labels:
        if len(label_text) > 2:  # Filter out single letters or very short text
            label_center_x, label_center_y = x + w // 2, y + h // 2
            label_right_x = x + w
            label_left_x = x
            label_bottom_y = y + h

            # Check if the label is close to any point of interest
            for (poi_x, poi_y, radius) in points_of_interest:
                # Check closeness for labels to the left or right of the point of interest
                if abs(label_right_x - poi_x) <= closeness_threshold or abs(label_left_x - poi_x) <= closeness_threshold:
                    if abs(label_center_y - poi_y) <= closeness_threshold:  # Ensure the label is vertically aligned
                        poi_labels.append(label_text)
                        break
                # Check closeness for labels underneath the point of interest
                if abs(label_bottom_y - poi_y) <= closeness_threshold and abs(label_center_x - poi_x) <= closeness_threshold:
                    poi_labels.append(label_text)
                    break
    print(f"Matched {len(poi_labels)} point of interest labels.")

    # Return the processed image and point of interest labels
    _, encoded_img = cv2.imencode('.png', image)
    return image, poi_labels

# Define the Gradio interface
def gradio_interface(image, closeness_threshold, label_distance_threshold):
    print("Gradio interface invoked.")
    return process_image(image, closeness_threshold, label_distance_threshold)

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(minimum=10, maximum=100, value=50, label="Closeness Threshold (pixels)"),
        gr.Slider(minimum=5, maximum=50, value=10, label="Label Distance Threshold (pixels)")
    ],
    outputs=[
        gr.Image(label="Processed Image with Circles"),
        gr.Textbox(label="Point of Interest Labels")
    ],
    title="Point of Interest Detection",
    description="Upload an image to detect points of interest and nearby labels. Adjust the closeness and label distance thresholds as needed."
)

# Launch the Gradio app
if __name__ == "__main__":
    print("Launching Gradio app...")
    interface.launch()
