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
import requests

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
from detectron2.data import DatasetCatalog
from datetime import datetime

from typing import Optional

from fastapi import FastAPI

app = FastAPI(debug=True)


def callback(id: int, count: int,degree: int, out):
    global URL
    print("send request to backend module")
    URL = "http://server.yesilkalacak.com/api/camera/addCameraReport"
    date_time = datetime.now()
    str_date_time = date_time.strftime("%d-%m-%Y-%H:%M:%S.%f")
    path = "/home/server/public_html/storage/app/public/stream/result/" + str(id);
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + "/" + str_date_time + ".png"
    out.save(filename)
    data = {
        'stream': id,
        'description': "detected {} instances".format(count),
        'path': "stream/result/" + str(id) + "/" + str_date_time + ".png",
        'degree': degree,
        'count': count
    }
    r = requests.post(url=URL, data=data)
    #print(r.status_code)
    #print(r.text)


@app.get("/stream/{id}")
def stream(id: int, path: str, showDetectWindwos: bool = False):
    return check_media(path, showDetectWindwos, id, callback)


@app.get("/check")
def check_media(path: str, showDetectWindwos=False, cameraID=None, callBack=None):
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    if "my_dataset_test" not in DatasetCatalog.list():
        register_coco_instances("my_dataset_test", {}, "content/test/_annotations.coco.json", "content/test")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.WEIGHTS = os.path.join("model_final.pth")

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # your number of classes + 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.DEVICE = "cpu"
    cfg.TEST.DETECTIONS_PER_IMAGE = 10
    cfg.INPUT.MIN_SIZE_TEST = 200
    cfg.INPUT.MAX_SIZE_TEST = 500
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    if ".jpg" in path:
        imageName = "/home/server/public_html/storage/app/public/" + path

        im = cv2.imread( imageName)

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
        out.save(imageName  + "RES.jpg")

        return {
            "detect": True if len(outputs["instances"]) > 0 else False,
            "time": took_time,
            "detect_fires": len(outputs["instances"]) if len(outputs["instances"]) > 0 else 0,
            "decree": int(outputs["instances"].scores[0] * 100) if len(outputs["instances"]) > 0 else 0
        }

    if cameraID:
        video = cv2.VideoCapture(path)
        frames_per_second = video.get(cv2.CAP_PROP_FPS) / 10
        video_visualizer = VideoVisualizer(test_metadata)

        count = 0
        found_counter = 0
        processed_counter = 0
        while video.isOpened():
            success, frame = video.read()
            if success:
                count = count + 1
                if count % 10 == 0:
                    processed_counter = processed_counter + 1
                    start_time = time.time()
                    outputs = predictor(frame)
                    logger.info(
                        "frame: {}:  -----  {} in {:.2f}s".format(
                            count,
                            "detected {} instances".format(len(outputs["instances"])),
                            time.time() - start_time,
                        )
                    )

                    if len(outputs["instances"]) > 0:
                        found_counter = found_counter + 1

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out = video_visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

                    if callback:
                        if len(outputs["instances"]) > 0:
                            callback(cameraID, len(outputs["instances"]),int(outputs["instances"].scores[0] * 100), out)

                    if (showDetectWindwos):
                        cv2.namedWindow("COCO detections", cv2.WINDOW_NORMAL)
                        cv2.imshow("COCO detections", out)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit
            else:
                break

        video.release()

        if (showDetectWindwos):
            cv2.destroyAllWindows()
        return {
            "all_frames": count,
            "processed_frames": processed_counter,
            "detect_fires": found_counter,
        }


    if ".mp4" in path:
        path = "/home/server/public_html/storage/app/public/" + path
        video_out_path =  path + "RES.mp4"

        video = cv2.VideoCapture(path)
        frames_per_second = video.get(cv2.CAP_PROP_FPS) / 10

        output_file = cv2.VideoWriter(filename=video_out_path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps=float(frames_per_second), frameSize=
                                      (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      isColor=True, )

        video_visualizer = VideoVisualizer(test_metadata)

        count = 0
        found_counter = 0
        processed_counter = 0
        decree = 0
        logs = []
        while video.isOpened():
            success, frame = video.read()
            if success:
                count = count + 1
                if count % 10 == 0:
                    processed_counter = processed_counter + 1
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
                        found_counter = found_counter + len(outputs["instances"])
                        if int(outputs["instances"].scores[0] * 100) > decree:
                            decree = int(outputs["instances"].scores[0] * 100)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out = video_visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

                    out = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
                    output_file.write(out)

            else:
                break

        video.release()
        output_file.release()
        cv2.destroyAllWindows()
        return {
            "all_frames": count,
            "processed_frames": processed_counter,
            "detect_fires": found_counter,
            "decree": decree,
            "logs": logs
        }

    return {
        "error": "file type"
    }
