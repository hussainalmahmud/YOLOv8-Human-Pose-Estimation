# YOLOv8-Human-Pose-Estimation 📚

Welcome to the YOLOv8-Human-Pose-Estimation Repository! 🌟 This project is dedicated to improving the prediction of the pre-trained YOLOv8l-pose model from Ultralytics. Here, you'll find scripts specifically written to address and mitigate common challenges like reducing False Positives, filling gaps in Missing Detections across consecutive frames, and stabilizing Detections in dynamic environments 🚀🔍.

## Installation 🛠️

### Option 1: Using pip or make 📦

```bash
pip install -r requirements.txt

# or 

make install
```

## Usage 🖥️

The repository includes various scripts, each serving a specific purpose in the realm of pose detection with YOLOv8:

### 1. `pose_predict.py` 🤖

- **Description**: Perform standard pose prediction using pre-trained YOLOv8 models.
- **Use Case**: Use this script to fine-tune the confidence threshold of pose detection for various input sources, including videos, images, or even real-time webcam feeds.
#### **Example Command**:
```
python pose_predict.py --model yolov8l-pose.pt --source 0 --is_video --view-img # webcam

python pose_predict.py --model yolov8l-pose.pt --source video.mp4 --is_video --save-img --view-img # Video source

python pose_predict.py --model yolov8l-pose.pt --source img_name.jpg --save-img --view-img # single image

python pose_predict.py --model yolov8l-pose.pt --source folder_name --save-img # folder of images

```

### 2. `pose_valid.py` 📊

- Description: Automates the evaluation of the YOLOv8 pose model across multiple confidence thresholds to determine the most effective setting.

- Use Case: Essential for optimizing model accuracy by identifying the ideal confidence threshold through systematic testing and metric analysis.

- **Features**:
  - 🎚 **Automated Threshold Testing**: Runs the model validation over a series of confidence thresholds ranging from 0.05 to 0.9.
  - 📈 **Performance Metrics Recording**: Collects and logs important metrics  like Precision (P), Recall (R), mean Average Precision (mAP), and F1 Score for each threshold inside a csv file.

**Example Command**:
  ```bash
  python pose_valid.py --model_file yolov8l-pose.yaml --weights yolov8l-pose.pt --dataset coco8-pose.yaml
  ```

### 3. `pose_fusion_predict.py` 🔮

- **Description**: Combines object tracking with an auxiliary segmentation network to enhance pose estimation results from the YOLOv8 model.

- **Use Case**: Ideal for scenarios where a single model's output needs refinement, especially in terms of accuracy and stability in pose detection.

**Note**: This script is still under development and serves as a prototype for the proposed method. It offers a glimpse into the fusion technique for improved pose estimation.

- **Example Command**:
  ```bash
  python pose_fusion_predict.py --pose_model yolov8l-pose.pt --seg_model yolov8l-seg.pt --source video.mp4 --is_video --save-img --view-img
  ```

### 4. `pose_custom_data_train.py` 🏋️‍♂️

- **Description**: Fine-tune the YOLOv8 pose detection model on a custom dataset. This process involves retraining the pre-trained model with data that's more specific to your task, enhancing model specificity and accuracy.

- **Use Case**: Optimal for scenarios requiring the model to adapt to unique environments or objects. Especially beneficial in improving detection precision and reducing false positives/negatives in challenging or borderline cases.

**Note**: Utilizing tools like Intel's CVAT for keypoint annotation can significantly enhance the effectiveness of the training process on custom datasets.

- **Example Command**:
  ```bash
  python pose_custom_data_train.py --model_file yolov8l-pose.yaml --weights yolov8l-pose.pt --dataset your_custom_dataset.yaml
  ```

### 5. `pose_custom_data_tune.py` 🔧

- **Description**: hyperparameter tuning of the YOLOv8 pose detection model using custom datasets. This script can help refines the model by adjusting specific parameters to align closely with the unique features of your data.

- **Use Case**: Ideal when seeking to enhance the precision and effectiveness of the YOLOv8 model for specific use cases, particularly where default settings might not yield optimal results.

**Note**: This script is designed to optimize the model's learning process through tailored hyperparameter adjustments, ensuring a more accurate and efficient performance on your custom data.

- **Example Command**:
  ```bash
  python pose_custom_data_tune.py --weights yolov8l-pose.pt --dataset your_custom_dataset.yaml
  ```