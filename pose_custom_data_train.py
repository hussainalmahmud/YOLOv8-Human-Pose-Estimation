from ultralytics import YOLO
import argparse


def train(model_file, weights, dataset):
    # Load a model
    model = YOLO(model_file).load(weights)  # build from YAML and transfer weights

    # Train the model
    model.train(
        model=model,
        data=dataset,  # specify our own custom dataset if needed
        epochs=10,
        patience=50,
        batch=16,
        imgsz=640,
        save=True,
        device="cpu",
        workers=8,
        pretrained=True,
        optimizer="auto",
        seed=0,
        resume=True,
        amp=True,
        fraction=1.0,
        lr0=0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="yolov8l-pose.yaml",
        help="initial yaml file path",
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8l-pose.pt", help="weights file path"
    )
    parser.add_argument(
        "--dataset", type=str, default="coco8-pose.yaml", help="initial yaml file path"
    )
    return parser.parse_args()


def main(local_opt):
    train(local_opt.model_file, local_opt.weights, local_opt.dataset)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
