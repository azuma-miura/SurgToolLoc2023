import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
import json

from mmdetection.mmdet.apis import init_detector, inference_detector
import mmcv
import os
import torch

####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = True


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )


class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                    
        
        # Specify the path to model config and checkpoint file
        os.chdir("./mmdetection")
        self.config_file = 'projects/configs/co_deformable_detr/myconfig.py'
        self.checkpoint_file = 'work_dirs/latest.pth'

        # build the model from a config file and a checkpoint file
        self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda' if torch.cuda.is_available() else 'cpu')
        os.chdir("..")
        
        
        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "force_bipolar",
                          "clip_applier",
                          "tip_up_fenestrated_grasper",
                          "cadiere_forceps",
                          "bipolar_forceps",
                          "vessel_sealer",
                          "suction_irrigator",
                          "bipolar_dissector",
                          "prograsp_forceps",
                          "stapler",
                          "permanent_cautery_hook_spatula",
                          "grasping_retractor"]

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def convert_float32(self, obj):
        if isinstance(obj, dict):
            return {key: self.convert_float32(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_float32(element) for element in obj]
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj

    def save(self):
        save_target = self.convert_float32(self._case_results[0])
        with open(str(self._output_file), "w") as f:
            json.dump(save_target, f)

    def generate_bbox(self, frame_id, results):
        # bbox coordinates are the four corners of a box: [x, y, 0.5]
        # Starting with top left as first corner, then following the clockwise sequence
        # origin is defined as the top left corner of the video frame
        predictions = []
        for result in results:
            label = result["label"]
            bbox = result["bbox"]
            score = result["score"]
            
            name = f'slice_nr_{frame_id}_' + label
            bbox = [[bbox[0], bbox[1], 0.5],
                    [bbox[2], bbox[1], 0.5],
                    [bbox[2], bbox[3], 0.5],
                    [bbox[0], bbox[3], 0.5]]
            prediction = {"corners": bbox, "name": name, "probability": float(score)}
            predictions.append(prediction)
        return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        
        all_frames_predicted_outputs = []
        for fid in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            result = inference_detector(self.model, frame)
            filtered_result = self.output_detections(result)
            
            tool_detections = self.generate_bbox(fid, filtered_result)
            all_frames_predicted_outputs += tool_detections

        return all_frames_predicted_outputs
    
    def output_detections(self, result, score_thr=0.2):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
                
        output_list = []
    
        if bboxes is not None:
            if score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]

            for i, bbox in enumerate(bboxes):
                label = labels[i]
                class_name = self.model.CLASSES[label] if self.model.CLASSES is not None else str(label)
                if class_name =="tip-up fenestrated grasper":
                    class_name = "tip up fenestrated grasper"
                score = bbox[-1] if bbox.shape[0] == 5 else None
                output_dict = {'label': class_name, 'bbox': bbox[:4].tolist(), 'score': score}
                output_list.append(output_dict)

        return output_list



if __name__ == "__main__":
    Surgtoolloc_det().process()
