import tensorflow as tf
import cv2
import os
from tqdm import tqdm

# Define paths
input_directory = 'C:/Users/Akshat Panchal/Desktop/Winter Project/waymo_2.0/tfrecords/'  # Path where your .tfrecord files are located
output_images_dir = 'C:/Users/Akshat Panchal/Desktop/Winter Project/waymo_2.0/output/images/'  # Directory to save extracted images
output_labels_dir = 'C:/Users/Akshat Panchal/Desktop/Winter Project/waymo_2.0/output/labels/'  # Directory to save YOLO label files

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Function to parse a single TFRecord entry
def parse_tfrecord_function(example_proto):
    # Define the feature schema based on your TFRecord structure
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        # Include any other features needed (e.g., labels if available)
    }

    # Parse the input `tf.train.Example` proto using the feature description
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
    height = parsed_features['image/height']
    width = parsed_features['image/width']
    xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])

    return image, xmin, xmax, ymin, ymax, height, width

# Check if the input directory exists
if not os.path.exists(input_directory):
    raise FileNotFoundError(f"The system cannot find the path specified: '{input_directory}'")

# Process each .tfrecord file
for tfrecord_file in tqdm(os.listdir(input_directory)):
    if tfrecord_file.endswith('.tfrecord'):
        # Create a TensorFlow dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(os.path.join(input_directory, tfrecord_file))
        
        for i, record in enumerate(dataset):
            try:
                image, xmin, xmax, ymin, ymax, height, width = parse_tfrecord_function(record)

                # Convert TensorFlow Tensors to NumPy arrays
                image_np = image.numpy()
                height = height.numpy()
                width = width.numpy()

                # Save image using OpenCV
                image_filename = f'image_{tfrecord_file}_{i}.jpg'
                cv2.imwrite(os.path.join(output_images_dir, image_filename), image_np)

                # Create YOLO annotation file
                label_filename = f'image_{tfrecord_file}_{i}.txt'
                with open(os.path.join(output_labels_dir, label_filename), 'w') as label_file:
                    for x_min, x_max, y_min, y_max in zip(xmin.numpy(), xmax.numpy(), ymin.numpy(), ymax.numpy()):
                        # Convert bounding box coordinates to YOLO format
                        x_center = ((x_min + x_max) / 2) * width
                        y_center = ((y_min + y_max) / 2) * height
                        bbox_width = (x_max - x_min) * width
                        bbox_height = (y_max - y_min) * height

                        # Normalize the coordinates (class ID is 0 as a placeholder)
                        label_file.write(f'0 {x_center/width:.6f} {y_center/height:.6f} {bbox_width/width:.6f} {bbox_height/height:.6f}\n')

            except Exception as e:
                print(f"Error processing record in {tfrecord_file}: {e}")

print("Conversion completed.")



# Process each file in the directory
total_files = len([f for f in os.listdir(json_directory) if f.endswith('.json')])
processed_files = 0

for file_name in os.listdir(json_directory):
    if file_name.endswith('.json'):
        processed_files += 1
        json_path = os.path.join(json_directory, file_name)

        # Open JSON label file
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {file_name}: {e}")
                continue

        # Find corresponding image file
        base_name = file_name.replace('.json', '')
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            possible_image_path = os.path.join(json_directory, base_name + ext)
            if os.path.exists(possible_image_path):
                image_path = possible_image_path
                break

        if not image_path:
            print(f"[{processed_files}/{total_files}] No corresponding image found for {file_name}, skipping...")
            continue

        # Open image to get dimensions
        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        annotations = []

        for label in data.get("labels", []):
            category = label["category"].lower()
            if category not in class_mapping:
                print(f"Unknown category: {category}, skipping label")
                continue  # Skip unknown categories

            class_id = class_mapping[category]

            # Extract bounding box coordinates (assume box3d_projected contains 4 points)
            try:
                box = label["box3d_projected"]
                x_min = box['bottom_left_front'][0]
                y_min = box['top_left_front'][1]
                x_max = box['bottom_right_front'][0]
                y_max = box['bottom_left_front'][1]
            except KeyError as e:
                print(f"Error extracting bounding box from {file_name}: {e}")
                continue  # Skip this label if bbox is malformed

            # Normalize bounding box
            normalized_bbox = normalize_bbox(x_min, y_min, x_max, y_max, image_width, image_height)

            if normalized_bbox is None:
                continue  # Skip invalid bounding boxes

            x_center, y_center, width, height = normalized_bbox

            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save the annotations to a .txt file
        if annotations:
            output_txt_path = os.path.join(json_directory, f"{base_name}.txt")
            with open(output_txt_path, "w") as txt_file:
                txt_file.write("\n".join(annotations))

            print(f"[{processed_files}/{total_files}] Processed file: {file_name}, saved as {base_name}.txt")
        else:
            print(f"[{processed_files}/{total_files}] No valid annotations for {file_name}, skipping...")

print("Processing completed for all JSON files!")