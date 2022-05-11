import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from predictor import VisualizationDemo

from typing import Optional

from fastapi import FastAPI

app = FastAPI(debug=True)

def callback(id: int, count: int, time: str):
    print("Sum = {}".format(count))

@app.get("/stream/{id}")
def stream(id: int, path: str, showDetectWindwos: bool = False):
    check_media(path, showDetectWindwos, callback)
    return {
        "result": "process done"
    }

@app.get("/check/{path}")
def check_media(path: str, showDetectWindwos=False, cameraID=0, callBack=None):
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    register_coco_instances("my_dataset_test", {}, "content/test/_annotations.coco.json", "content/test")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.WEIGHTS = os.path.join("model_final.pth")

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # your number of classes + 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.INPUT.MIN_SIZE_TEST = 0
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    if ".jpg" in path:
        imageName = path
        im = cv2.imread('content/input/' + imageName)
        start_time = time.time()
        outputs = predictor(im)
        logger.info("{}:  -----  {} in {:.2f}s".format(
            imageName,
            "detected {} instances".format(len(outputs["instances"])),
            time.time() - start_time,
        ))
        took_time = time.time() - start_time

        v = Visualizer(im[:, :, ::-1], metadata=test_metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out.save(os.path.join("content/out/", imageName))

        return {
            "detect": True if len(outputs["instances"]) > 0 else False,
            "count": len(outputs["instances"]),
            "time": took_time,
        }

    if ".mp4" in path:
        video_path = path
        video = cv2.VideoCapture('content/input/' + video_path)
        frames_per_second = video.get(cv2.CAP_PROP_FPS) / 10
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / 10
        output_fname = os.path.splitext(os.path.join("content/out/", video_path))[0] + ".mp4"

        output_file = cv2.VideoWriter(filename=output_fname, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps=float(frames_per_second), frameSize=(
                int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      isColor=True, )

        video_visualizer = VideoVisualizer(test_metadata)

        count = 0
        found_counter = 0
        logs = []
        while video.isOpened():
            success, frame = video.read()
            count = count + 1
            if success:
                if count % 10 == 0:
                    start_time = time.time()
                    outputs = predictor(frame)
                    logger.info(
                        "frame: {}:  -----  {} in {:.2f}s".format(
                            count,
                            "detected {} instances".format(len(outputs["instances"])),
                            time.time() - start_time,
                        )
                    )
                    logs.append("frame: {}:  -----  {} in {:.2f}s".format(
                        count,
                        "detected {} instances".format(len(outputs["instances"])),
                        time.time() - start_time,
                    ))
                    if len(outputs["instances"]) > 0:
                        found_counter = found_counter + 1

                    if callback:
                        callback(cameraID, len(outputs["instances"]), time.time())

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out = video_visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
                    out = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)

                    output_file.write(out)
                    if (showDetectWindwos):
                        cv2.namedWindow("COCO detections", cv2.WINDOW_NORMAL)
                        cv2.imshow("COCO detections", out)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit

                    if found_counter > 10:
                        video.release()
                        output_file.release()
                        if (showDetectWindwos):
                            cv2.destroyAllWindows()
                        return {
                            "all_frames": num_frames * 10,
                            "processed_frames": count,
                            "check_frames": count / 10,
                            "detect_frames": found_counter,
                            "logs": logs
                        }
            else:
                break

        video.release()
        output_file.release()
        if (showDetectWindwos):
            cv2.destroyAllWindows()
        return {
            "all_frames": num_frames * 10,
            "processed_frames": count,
            "check_frames": count / 10,
            "detect_frames": found_counter,
            "logs": logs
        }
    return {
        "error": "file type"
    }
