from ultralytics import YOLO
import argparse
import csv
from matplotlib import pyplot as plt


def validate(model_file, weights, dataset):
    model = YOLO(model_file).load(weights)  # build from YAML and transfer weights
    conf_thresholds = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
    ]
    f1_scores = []

    # with open("validate_results.csv", mode="w", newline="") as file:
    with open("validate_results.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Conf.",
                "Box P",
                "Box R",
                "Box mAP50",
                "Box mAP50-95",
                "Box F1",
                "Pose P",
                "Pose R",
                "Pose mAP50",
                "Pose mAP50-95",
                "Pose F1",
            ]
        )

        for conf in conf_thresholds:
            metrics = model.val(
                data=dataset,
                conf=conf,
            )
            f1_score = metrics.pose.f1[0]
            f1_scores.append(f1_score)

            writer.writerow(
                [
                    conf,
                    metrics.box.p[0],
                    metrics.box.r[0],
                    metrics.box.map50,
                    metrics.box.map,
                    metrics.box.f1[0],
                    metrics.pose.p[0],
                    metrics.pose.r[0],
                    metrics.pose.map50,
                    metrics.pose.map,
                    metrics.pose.f1[0],
                ]
            )
    # Plotting
    plt.plot(conf_thresholds, f1_scores, marker="o")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Confidence Threshold")
    plt.grid(True)
    plt.savefig("f1_scores_plot.png")
    plt.show()


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
    validate(local_opt.model_file, local_opt.weights, local_opt.dataset)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
