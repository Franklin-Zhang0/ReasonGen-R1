import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder_path",
    type = str,
    default = "/home/aiscuser/project/T2I-CompBench/examples/outputs/Janus-Pro-7B",
    required=True,
)
args = parser.parse_args()

folder = args.folder_path
setting_dict={
    "color" : "color_val/annotation_blip/blip_vqa_score.txt",
    "shape" : "shape_val/annotation_blip/blip_vqa_score.txt",
    "texture" : "texture_val/annotation_blip/blip_vqa_score.txt",
    "non_spatial" : "non_spatial_val/annotation_clip/score_avg.txt",
    "spatial" : "spatial_val/labels/annotation_obj_detection_2d/avg_score.txt",
    "numeracy" : "numeracy_val/annotation_num/score.txt",
    "complex" : "complex_val/annotation_3_in_1/vqa_score.txt",
}

import os
for key in setting_dict.keys():
    try:
        with open(os.path.join(folder, setting_dict[key]), 'r') as f:
            lines = f.readline()
            score = lines.strip().split(':')[-1]
            score = float(score)
            print(f"{key} score: {score}")
    except Exception as e:
        print(f"{key} score: None")