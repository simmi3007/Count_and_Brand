import streamlit as st
import yaml
import numpy as np
import cv2
from datetime import datetime
import openpyxl
from ultralytics import YOLO
import os

# Load the YOLOv8 model globally
model = YOLO('Brand_best.pt')

def start_counting():
    start_time = datetime.now()
    last_time = datetime.now()

    ct = 0
    total_output = 0
    ppm = 0
    ppm_average = 0

    rec_qty = 8
    qty = 0

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("datetime", "total_output", "minute", "average ppm", "ct", "ppm"))

    fn_yaml = r"datasets\\area.yml"
    config = {
        'save_video': False,
        'text_overlay': True,
        'object_overlay': True,
        'object_id_overlay': False,
        'object_detection': True,
        'min_area_motion_contour': 60,
        'park_sec_to_wait': 0.001,
        'start_frame': 0
    }

    cap = cv2.VideoCapture(0)

    with open(fn_yaml, 'r') as stream:
        object_area_data = yaml.safe_load(stream)

    object_status = [False] * len(object_area_data)
    object_buffer = [None] * len(object_area_data)

    st.write("Counting objects...")
    frame_placeholder = st.empty()

    while cap.isOpened():
        if st.session_state.stop_counting:
            break

        try:
            video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ret, frame = cap.read()
            if not ret:
                st.error("Capture Error")
                break

            frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            frame_out = frame.copy()

            if config['object_detection']:
                for ind, park in enumerate(object_area_data):
                    points = np.array(park['points'])
                    rect = cv2.boundingRect(points)
                    roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]

                    points[:, 0] -= rect[0]
                    points[:, 1] -= rect[1]

                    status = np.std(roi_gray) < 20 and np.mean(roi_gray) > 56

                    if status != object_status[ind] and object_buffer[ind] is None:
                        object_buffer[ind] = video_cur_pos
                    elif status != object_status[ind] and object_buffer[ind] is not None:
                        if video_cur_pos - object_buffer[ind] > config['park_sec_to_wait']:
                            if not status:
                                qty += 1
                                total_output += 1
                                st.session_state.total_count = total_output

                                current_time = datetime.now()
                                diff = current_time - last_time
                                ct = diff.total_seconds()
                                ppm = round(60 / ct, 2)
                                last_time = current_time

                                diff = current_time - start_time
                                minutes = diff.total_seconds() / 60
                                ppm_average = round(total_output / minutes, 2)

                                data = (current_time, total_output, minutes, ppm_average, ct, ppm)
                                ws.append(data)

                                if qty > rec_qty:
                                    ws.append(data)
                                    qty = 0

                            object_status[ind] = status
                            object_buffer[ind] = None
                    elif status == object_status[ind] and object_buffer[ind] is not None:
                        object_buffer[ind] = None

            if config['object_overlay']:
                for ind, park in enumerate(object_area_data):
                    points = np.array(park['points'])
                    color = (0, 255, 0) if object_status[ind] else (0, 0, 255)
                    cv2.drawContours(frame_out, [points], -1, color, 2)

            if config['text_overlay']:
                cv2.rectangle(frame_out, (1, 5), (350, 70), (0, 255, 0), 2)
                cv2.putText(frame_out, f"Total Output: {total_output}", (5, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            st.error(f"An error occurred: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return st.session_state.total_count

def brand_recognition(image_data):
    """Recognizes brands and displays bounding boxes on the image."""
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    results = model(image, conf=0.25)

    detected_brands = []

    for result in results:
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                brand_name = result.names[class_id]
                detected_brands.append(brand_name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, brand_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Processed Image with Detected Brands", use_column_width=True)

    st.write("Detected Brands:")
    if detected_brands:
        for brand in detected_brands:
            st.write(f"- {brand}")
    else:
        st.write("No brands detected.")

def main():
    st.title("Real-Time Inventory Product Counting")

    if "stop_counting" not in st.session_state:
        st.session_state.stop_counting = False
    if "total_count" not in st.session_state:
        st.session_state.total_count = 0

    if st.button("❌Stop count"):
        st.session_state.stop_counting = True
        st.write(f"Counting Stopped! Total Count: {st.session_state.total_count}")

    if st.button("Start Counting"):
        st.session_state.stop_counting = False
        with st.spinner("Counting in progress..."):
            total_count = start_counting()
            st.success("Counting completed!")
            st.write(f"Total Count: {total_count}")

    st.title("Brand Recognition")

    predefined_images = ["test images\\1.jpg", "test images\\2.jpg", "test images\\3.jpg"]
    selected_image = st.selectbox("Select Images for Trial", predefined_images)

    if st.button("Brand Recognition"):
        brand_recognition(open(selected_image, "rb").read())

    uploaded_file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        if st.button("Start Brand Recognition"):
            brand_recognition(uploaded_file.read())

if __name__ == "__main__":
    main()
