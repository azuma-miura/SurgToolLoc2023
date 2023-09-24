from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import cv2
import random

def output_detections(model, result, score_thr=0.5):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels is None or labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or labels is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    
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
            class_name = model.CLASSES[label] if model.CLASSES is not None else str(label)
            if class_name =="tip-up fenestrated grasper":
                class_name = "tip up fenestrated grasper"
            score = bbox[-1] if bbox.shape[0] == 5 else None
            output_dict = {'label': class_name, 'bbox': bbox[:4].tolist(), 'score': score}
            output_list.append(output_dict)

    return output_list

def set_up():
    # Specify the path to model config and checkpoint file
    config_file = 'projects/configs/co_deformable_detr/myconfig.py'
    checkpoint_file = 'work_dirs/latest.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    return model

def inference(video_path, model, score_thr=0.5):
    # test a single image and show the results
    # img = 'hdd/imgs/clip_000148_66.jpg'  # or img = mmcv.imread(img), which will only load it once
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("No")
    result = inference_detector(model, frame)
    # visualize the results in a new window
    # model.show_result(img, result)
    model.show_result(frame, result, score_thr=score_thr,
                      out_file=f'inference_results/result_{video_path.replace("mp4","").split("/")[-1]}.jpg')
    
    cap.release()

    # bboxは左上と右下の座標    
    return output_detections(model, result, score_thr)

def main():
    model = set_up()
    for _ in range(100):
        video_path = f'hdd/training_data/clip_{random.randint(0, 22000):06d}.mp4'
        result = inference(video_path, model, score_thr=0.4)
        print(result)

if __name__ == "__main__":
    main()