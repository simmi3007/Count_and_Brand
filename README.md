
# Real-Time Inventory Product Counting and Brand Recognition

This project implements a web-based application that performs real-time inventory product counting from video streams and brand recognition from images using the YOLOv8 object detection model. The application is built using *Streamlit* for the frontend interface and OpenCV for image processing.

## Features

- *Real-Time Product Counting*:
  - Upload a video file or choose from predefined trial videos.
  - Detect and count objects passing through a defined area in the video stream.
  - Real-time display of video frames with object detection overlay and updated object counts.
  - Performance statistics include the total output, average objects per minute, and real-time processing updates.
  
- *Brand Recognition*:
  - Upload images for brand recognition.
  - Detect logos/brands in the image using the YOLOv8 model and display bounding boxes around detected logos.
  - Provides a list of all detected brands in the image.

## Installation

### 1. Clone the repository

1. git clone https://github.com/Priya-161/Count_and_Brand.git
   cd Count_and_Brand
2. Set up a virtual environment (optional but recommended)
3. Install the required dependencies: pip install -r requirements.txt
4. Download the YOLOv8 model
Download the pre-trained YOLOv8 model file Brand_best.pt and place it in the root of the project directory. You can train your own YOLOv8 model or use an existing one. Ensure that the model is compatible with your application.

Usage
1. Running the Application
Start the Streamlit application using the following command:streamlit run app.py
2. Using the Application

Real-Time Product Counting
Upload a video file: Upload a .mp4, .avi, or .mov video file for product counting.
Select a trial video: Choose from the available trial videos for testing.
The application will process the video, count the products passing through the detection area, and display real-time updates. The final output will include the total object count and relevant performance statistics.

Brand Recognition
Upload an image: Upload an image in .jpg, .jpeg, or .png format for brand/logo recognition.
Select a trial image: Choose from the available sample images for testing.
The application will detect brands in the image using the YOLOv8 model and display the bounding boxes around detected brands. A list of all detected brands will also be displayed.
The detection area for object counting is defined in the area.yml file. You can modify the object detection areas by updating the coordinates of the detection regions.
Customization of the bounding area for product detection in the video stream is defined at the time of production as required.


