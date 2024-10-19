from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('Brand_best.pt')
results = model('4.jpg', conf=0.25)  # Adjust the confidence threshold


# Display predictions on the image(s)
for result in results:
    result.show()

# Optional: Save the result image(s)
for idx, result in enumerate(results):
    # Construct a filename for saving
    save_path = f'inference_result_{idx}.jpg'
    result.save(save_path) 