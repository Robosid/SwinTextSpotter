# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from threading import Thread
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo


# constants
WINDOW_NAME = "COCO detections"


class ThreadedCamera(object):
    def __init__(self, src=""):   #if a new object is made, this --init-- is executed once
        self.capture = cv2.VideoCapture(src)   #cv2.VideoCapture object (has a method "read") . the argument is "src" (this is the address)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 1000
        self.FPS_Display = 50 #frequency for display on laptop

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())  #we make a new thread
        self.thread.daemon = True     #daemon: runs in the background
        self.thread.start()       #starts the thread and calls self.update
        self.frame_time_pre = time.time()
        self.frame_time_post = time.time()
        self.center = (0, 0)
        self.pitch = 0
        self.pan = 0
        self.tag_detected = False

    def update(self):   #the self.update ,  this loop performs the querries on the server
        i = 0  #couner
        k = 0  #image frame number
        while True:   #runs the whole time
            if self.capture.isOpened():
                buffer_time = time.time()
                (self.status, self.frame) = self.capture.read()   #in self.status and .frame the results are stored
                self.frame_time_pre = buffer_time
                self.frame_time_post = time.time()
                j = i % 30  # should count from zero to 9 all the times
                if j == 1:
                   # cv2.imwrite('Frame' + str(k) + '.jpg', self.frame)
                    k += 1
                   # print(f"image {k} stored")
                i += 1
            # time.sleep(self.FPS)

    def show_frame(self):
        #self.add_aruco_tag()
        cv2.imshow('frame', self.frame)

        #print(self.frame.nbytes)
        #print(f"T_prev: {round(self.frame_time_pre,3)}, "
        #      f"Delta_post: {round(self.frame_time_post-self.frame_time_pre,3)},"
        #      f" Delta_cur: {round(time.time()-self.frame_time_pre,3)}"
        #      f" Delta_cur+sampling: {round(time.time()-self.frame_time_pre+self.FPS,3)}")
        cv2.waitKey(int(1000/self.FPS_Display))


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

    # -----
    from projects.SWINTS.swints import add_SWINTS_config
    add_SWINTS_config(cfg)
    # -----

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    hh = []
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold, path)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        src = 'rtmp://rtmp.airobot.eu/live/Mri9wy'
        #cam =  ThreadedCamera(src)
        cam = cv2.VideoCapture(src) 
        for vis in tqdm.tqdm(demo.run_on_video(cam, args.confidence_threshold)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # x264
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video, args.confidence_threshold), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
