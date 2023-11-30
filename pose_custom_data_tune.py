from ultralytics import YOLO
import argparse


def tune(weights, dataset):
    # Load a model
    model = YOLO(weights) 

    # Train the model
    model.tune(data=dataset, epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,default='yolov8l-pose.pt' , help='weights file path')
    parser.add_argument('--dataset', type=str, default='coco8-pose.yaml', help='initial yaml file path')
    opt = parser.parse_args()
    return opt

def main(opt):
    tune(opt.weights, opt.dataset)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)