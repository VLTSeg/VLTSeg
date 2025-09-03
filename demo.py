import argparse
import torch
from torchvision.io import read_image, ImageReadMode
from mmseg.apis import init_model, inference_model, show_result_pyplot
import vltseg

parser = argparse.ArgumentParser(description="Run demo display with a specified checkpoint path.")
parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file.")
args = parser.parse_args()

# Use the provided checkpoint path
checkpoint_path = args.checkpoint_path

# Since we are only running a single manual inference, the exact config we choose does not matter.
# If you want to test one of the checkpoints trained on real data, use this one:
config_path = "configs/mask2former_evaclip_2xb4_20k_cityscapes2cityscapes.py"
# If you want to test one of the checkpoints trained on synthetic data, use the this one instead:
# config_path = "configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py"


device = "cuda" if torch.cuda.is_available() else "cpu"
# This is a Cityscapes image, more specifically, frankfurt_000001_055709
img_path = "images/cityscapes_frankfurt.png"
gt_exists = True
gt_path = "images/cityscapes_frankfurt_gt.png"
output_name = "cityscapes_frankfurt"

model = init_model(config_path, checkpoint_path, device)
result = inference_model(model, img_path)

if gt_exists:
    pred_data = result.pred_sem_seg.data
    gt_data = read_image(gt_path).to(pred_data.dtype).to(device)
    pred_data[gt_data==255] = 255

vis_image = show_result_pyplot(model, img_path, result, opacity=1.0, with_labels=False,
                               show=False, save_dir="images",
                               out_file=f"images/{output_name}_pred_color.png")
vis_image_labeled = show_result_pyplot(model, img_path, result, opacity=1.0, with_labels=True,
                                       show=False, save_dir="images",
                                       out_file=f"images/{output_name}_pred_color_labeled.png")

if gt_exists:
    result.pred_sem_seg.data = gt_data

    vis_image = show_result_pyplot(model, img_path, result, opacity=1.0, with_labels=False,
                                   show=False, save_dir="images",
                                   out_file=f"images/{output_name}_gt_color.png")
    vis_image_labeled = show_result_pyplot(model, img_path, result, opacity=1.0, with_labels=True,
                                           show=False, save_dir="images",
                                           out_file=f"images/{output_name}_gt_color_labeled.png")

