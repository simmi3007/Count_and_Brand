
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

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
2. Set up a virtual environment (optional but recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate        # For Linux/Mac
venv\Scripts\activate           # For Windows
3. Install the required dependencies
bash
Copy code
pip install -r requirements.txt
4. Download the YOLOv8 model
Download the pre-trained YOLOv8 model file Brand_best.pt and place it in the root of the project directory. You can train your own YOLOv8 model or use an existing one. Ensure that the model is compatible with your application.
Usage
1. Running the Application
Start the Streamlit application using the following command:

bash
Copy code
streamlit run app.py
2. Access the Application
Open a web browser and navigate to:

arduino
Copy code
http://localhost:8501/
3. Using the Application
Real-Time Product Counting
Upload a video file: Upload a .mp4, .avi, or .mov video file for product counting.
Select a trial video: Choose from the available trial videos for testing.
The application will process the video, count the products passing through the detection area, and display real-time updates. The final output will include the total object count and relevant performance statistics.

Brand Recognition
Upload an image: Upload an image in .jpg, .jpeg, or .png format for brand/logo recognition.
Select a trial image: Choose from the available sample images for testing.
The application will detect brands in the image using the YOLOv8 model and display the bounding boxes around detected brands. A list of all detected brands will also be displayed.

Project Structure
plaintext
Copy code
your-repo-name/
│
├── app.py                  # Main Streamlit app file
├── Brand_best.pt            # YOLOv8 model file
├── area.yml                 # Configuration for detection areas
├── requirements.txt         # List of Python dependencies
├── README.md                # Project documentation
├── trial_videos/            # Sample videos for testing
│   ├── a.mp4
│   ├── b.mp4
│   ├── c.mp4
│   └── d.mp4
├── images/                  # Predefined images for brand recognition
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
Configuration
The detection area for object counting is defined in the area.yml file. You can modify the object detection areas by updating the coordinates of the detection regions.

Example area.yml structure:

yaml
Copy code
object_area_data:
  - points: 
    - [x1, y1]
    - [x2, y2]
    - [x3, y3]
    - [x4, y4]
This allows customization of the bounding area for product detection in the video stream.

Dependencies
Streamlit: Frontend framework for building web apps with Python.
OpenCV: Library for computer vision tasks, including frame processing.
YOLOv8: Object detection model for detecting objects and brands in images.
NumPy: For array and numerical operations.
PyYAML: For reading configuration from YAML files.
Openpyxl: For writing performance statistics to Excel files.
Install these dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Example Outputs
Real-Time Product Counting
As the video is processed, objects passing through a predefined region of interest will be counted, and you’ll see a real-time overlay of the detected objects:


Brand Recognition
When an image is uploaded for brand recognition, the application will display bounding boxes around detected logos and list the detected brands:


License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Streamlit - for providing an easy-to-use framework for web-based applications.
YOLOv8 - for the powerful object detection model.
OpenCV - for image processing and video frame manipulation.
markdown
Copy code

### Notes:
- Replace the placeholder link in the clone command with your repository URL (https://github.com/yourusername/your-repo-name.git).
- Make sure to include example images for sample_counting_output.png and sample_brand_recognition_output.png in the images/ folder to illustrate the functionality.
