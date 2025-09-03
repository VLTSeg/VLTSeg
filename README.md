## VLTSeg: A Domain Generalized Semantic Segmentation Model

**[ACCV2024 Paper](https://arxiv.org/pdf/2312.02021v3)**

This repo implements VLTSeg, the semantic segmentation model from the paper [Strong but Simple: A Baseline for Domain Generalized Dense Perception by CLIP-based Transfer Learning](https://vltseg.github.io/) using the open-source framework [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Note that we are not the authors of the paper. This reimplementation merely aims to remedy the fact that the code for VLTSeg is not currently publicly available. You can use the repo to [train your own version of the model](#training--testing-the-model) or use one of the [pretrained checkpoints](#checkpoints) available via Zenodo.

## Setting Up Your Virtual Environment

To run this repository, you will need a machine with GCC 9.2.0, Python 3.8.x (we used 3.8.20) and a GPU compatbile with CUDA 12.1. 

We recommend setting up a new virtual environment, following the steps below. We will assume you wish to install the repo under `/PROJECTS/VLTSeg`, keep your checkpoints under `/CHECKPOINTS` and keep your datasets under `/DATA`, so change all relevant paths to fit your workspace as required.

```
### Create and activate the virtual environment
conda deactivate
conda clean --all
conda create --name vltseg python=3.8 -y
conda activate vltseg

### Clone VLTSeg
cd /PROJECTS
git clone https://github.com/VLTSeg/VLTSeg.git VLTSeg

### Install requirements
pip install torch==2.3.0 torchvision xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim==0.3.9
mim install mmengine==0.10.5
mim install mmcv==2.1.0
pip install "mmdet>=3.0.0rc4"
pip install -r VLTSeg/requirements.txt

### Clone and install MMSegmentation from source
git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
mim install -e .
cd ..

### Install VLTSeg from source
cd VLTSeg
pip install -e .
ln -s /CHECKPOINTS /PROJECTS/VLTSeg/checkpoints
ln -s /DATA /PROJECTS/VLTSeg/data

### Download the pretrained EVA-CLIP weights
wget https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_psz14_s4B.pt -P checkpoints

### Download any of the VLTSeg checkpoints linked at the bottom of this page and save them to the checkpoints folder

### Test the installation
# This script runs inference on images/cityscapes_frankfurt.png and saves the output to the images folder.
# To run this script, you need at least one of the checkpoints linked at the bottom of this page.
# If you use a synthetic2real checkpoint (like GTA_1), modify the code to use the corresponding config.
python demo.py "checkpoints/vltseg_checkpoint_mapillary+cityscapes_2.pth"
```

The symbolic links to `/CHECKPOINTS` and `/DATA` are required so that the config files can work with relative paths and find your checkpoints and datasets respectively, without modification. If you would rather avoid creating symbolic links, you need to modify the `pretrained` key in [`configs/_base_/models/eva-clip+mask2former.py`](configs/_base_/models/eva-clip+mask2former.py), as well as all `data_root` keys in all files under [`configs/_base_/datasets`](configs/_base_/datasets).

## Preparing the Datasets

To train the model, you will need to download the relevant datasets and, in most cases, reformat them to be compatible with MMSegmentation and the Cityscapes labels. To this end, we use the excellent guidance and converter scripts from the [HRDA repository](https://github.com/lhoyer/HRDA/tree/master?tab=readme-ov-file#setup-datasets). 

We have copied the scripts into [`tools/dataset_converters`](tools/dataset_converters) and use them in the following instructions. Please note that the converter scripts make use of various IO helper methods defined in an [older version of MMCV](https://github.com/open-mmlab/mmcv/tree/v1.3.7), that are no longer present in the latest version. As such, **you will need to set up a separate virtual environment to run the scripts**. You can find the instructions [here](tools/dataset_converters/README.md). Once you have done so, continue with the steps below.

```
### Activate the prepared virtual environment
conda deactivate
conda activate hrda
cd /DATA

### GTA
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/[ID]_images.zip   # where [ID] ranges from 01 to 10
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/[ID]_labels.zip   # where [ID] ranges from 01 to 10
unzip [FILENAME] -d gta                                                             # where [FILENAME] ranges over all 20 files
python /PROJECTS/VLTSeg/tools/dataset_converters/gta.py /DATA/gta --nproc 8

### SYNTHIA
wget --no-check-certificate http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar
unrar x SYNTHIA_RAND_CITYSCAPES.rar
mv RAND_CITYSCAPES synthia
python /PROJECTS/VLTSeg/tools/dataset_converters/synthia.py /DATA/synthia --nproc 8
```

It is worth pointing out that the unrar process for `SYNTHIA_RAND_CITYSCAPES.rar` may leave some of the ground truth images corrupted. In our case, this affected seven files, even after repeated attempts. If you find that the converter script fails to generate a matching `_labelTrainIds.png` for some of the files in `/DATA/synthia/GT/LABELS`, consider simply deleting the affected training images in both `/DATA/synthia/RGB` and `/DATA/synthia/GT/LABELS`. If you do not, the training script will stumble over the images with missing ground truth eventually, and crash.

```
### Cityscapes
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=[USERNAME]&password=[PASSWORD]&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip [FILENAME] -d cityscapes                                                      # where [FILENAME] ranges over both files
python /PROJECTS/VLTSeg/tools/dataset_converters/cityscapes.py /DATA/cityscapes --nproc 8

### ACDC
# Manually download rgb_anon_trainvaltest.zip and gt_trainval.zip from https://acdc.vision.ee.ethz.ch/download
unzip rgb_anon_trainvaltest.zip -d /DATA/acdc
unzip gt_trainval.zip -d /DATA/acdc
# These five commands flatten the folder structure
rsync -a /DATA/acdc/rgb_anon/*/train/*/* /DATA/acdc/rgb_anon/train/
rsync -a /DATA/acdc/rgb_anon/*/val/*/* /DATA/acdc/rgb_anon/val/
rsync -a /DATA/acdc/rgb_anon/*/test/*/* /DATA/acdc/rgb_anon/test/
rsync -a /DATA/acdc/gt/*/train/*/*_labelTrainIds.png /DATA/acdc/gt/train/
rsync -a /DATA/acdc/gt/*/val/*/*_labelTrainIds.png /DATA/acdc/gt/val/
# These three commands create empty ground truth files for the test set images
rsync -a /DATA/acdc/rgb_anon/test/* /DATA/acdc/gt/test/
for f in /DATA/acdc/gt/test/*_rgb_anon.png; do mv "$f" "$(echo "$f" | sed s/_rgb_anon/_gt_labelTrainIds/)"; done
for f in /DATA/acdc/gt/test/*_gt_labelTrainIds.png; do scp /PROJECTS/VLTSeg/tools/dataset_converters/acdc_white.png "$f"; done

### BDD100k
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip
wget https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip
unzip [FILENAME]                                                                    # where [FILENAME] ranges over all 4 files

### Mapillary
# Manually download v1.2 from https://www.mapillary.com/dataset/vistas
unzip mapillary-vistas-dataset_public_v1.2.zip -d /DATA/mapillary
python /PROJECTS/VLTSeg/tools/dataset_converters/mapillary.py /DATA/mapillary/training --nproc 8
python /PROJECTS/VLTSeg/tools/dataset_converters/mapillary.py /DATA/mapillary/validation --nproc 8
```

## Training & Testing the Model

### A Training Most Simple

Training the model is simple, once you have activated the virtual environment and set your working directory to the root of the repo:

```
conda activate vltseg
cd /PROJECTS/VLTSeg
sh tools/dist_train.sh [CONFIG] [NUM_GPUS] --work-dir [WORK_DIR]
```

Replace `[CONFIG]` by one of the configs located at the top level under [`configs`](configs). Each of these inherits from several of the files in [`configs/_base_`](configs/_base_). Making your own config is as simple as copying an existing config and changing either the key `_base_` or one of the few others at the top of the file.

For example, if we wanted to train for 5,000 iterations, at a batch size of 16, on 2 GPUs, on the GTA dataset, validating on the Cityscapes dataset, we would run

```
sh tools/dist_train.sh configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py 2 --work-dir /WORK_DIR
```

### Pretraining with a Frozen Backbone

While this is not part of the original paper, we have occasionally observed slightly better results in our own experiments by first pretraining for 1,000 iterations with all backbone weights frozen. To do so, use [`configs/_base_/schedules/schedule_1k_frozen.py`](configs/_base_/schedules/schedule_1k_frozen.py) like so:

```
sh tools/dist_train.sh configs/mask2former_evaclip_2xb8_1k_frozen_gta2cityscapes.py 2 --work-dir /WORK_DIR_PRETRAIN
```

Once the pretraining has concluded, you can start the actual training from that checkpoint by adding `--cfg-options load_from=[CHECKPOINT]` like this:

```
sh tools/dist_train.sh configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py 2 --work-dir /WORK_DIR --cfg-options load_from="/WORK_DIR_PRETRAIN/iter_1000-????????.pth"
```

### Testing a Trained Model

Testing functions much the same as training, except specifying a checkpoint is no longer optional:

```
sh tools/dist_test.sh [CONFIG] [CHECKPOINT] [NUM_GPUS] --work-dir [WORK_DIR]
```

For example, if we wanted to evaluate how well our earlier model generalizes to ACDC, we would run

```
sh tools/dist_test.sh configs/mask2former_evaclip_2xb8_5k_gta2acdc.py /WORK_DIR/iter_5000-????????.pth 2 --work-dir /WORK_DIR
```

If you want to employ test time augmentations, simply add the flag `--tta` at the end of the command. All configs come with multi-scale evaluation and a (not so random) random fip by default. This option was used for the Cityscapes and ACDC test set benchmarks.

### Training & Testing on a Slurm Cluster

If you are working on a computing cluster running Slurm, we have provided examples of `.sbatch` scripts, which contain all the same options discussed above, namely [`tools/slurm_train.sbatch`](tools/slurm_train.sbatch) and [`tools/slurm_test.sbatch`](tools/slurm_test.sbatch). In that case you would simply run

```
sbatch slurm_train.sbatch
```

## Checkpoints

You can find the following checkpoints, trained on various synthetic and real world datasets, on [Zenodo](https://zenodo.org/records/14766160).

| Checkpoint                                                                                                                  | Iterations Trained | Batch Size | `drop_path_rate` | ACDC(val) | ACDC(test) | BDD100K(val) | Cityscapes(val) | Cityscapes(test) | Mapillary(val) |
|:---------------------------------------------------------------------------------------------------------------------------:|:------------------:|:----------:|:----------------:|:---------:|:----------:|:------------:|:---------------:|:----------------:|:--------------:|
| [GTA_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_gta_1.pth?download=1)                                   | 5K                 | 16         | 0.20             | 62.34     | -          | 59.56        | 65.23           | -                | 66.07          |
| [GTA_2](https://zenodo.org/records/14766160/files/vltseg_checkpoint_gta_2.pth?download=1)                                   | 1K + 5K            | 8          | 0.15             | 60.12     | -          | 60.16        | 66.69           | -                | 66.49          |
| [SYNTHIA_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_synthia_1.pth?download=1)                           | 5K                 | 16         | 0.20             | 49.88     | -          | 50.66        | 56.85           | -                | 55.96          |
| [SYNTHIA_2](https://zenodo.org/records/14766160/files/vltseg_checkpoint_synthia_2.pth?download=1)                           | 1K + 5K            | 8          | 0.15             | 49.42     | -          | 52.13        | 57.55           | -                | 55.92          |
| [ACDC_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_acdc_1.pth?download=1)                                 | 20K                | 8          | 0.20             | 81.44     | -          | 65.85        | 79.51           | -                | 75.56          |
| [BDD100K_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_bdd100k_1.pth?download=1)                           | 20K                | 8          | 0.20             | 72.05     | -          | 71.43        | 77.74           | -                | 76.13          |
| [Cityscapes_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_cityscapes_1.pth?download=1)                     | 20K                | 8          | 0.20             | 73.16     | 77.28\*    | 65.10        | 84.83           | -                | 76.81          |
| [Cityscapes_2](https://zenodo.org/records/14766160/files/vltseg_checkpoint_cityscapes_2.pth?download=1)                     | 1K + 20K           | 8          | 0.15             | 73.92     | 77.11\*    | 65.84        | 85.60           | -                | 77.23          |
| [Mapillary+Cityscapes_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_mapillary+cityscapes_1.pth?download=1) | 20K + 40K          | 8          | 0.20             | -         | -          | -            | 86.30           | 86.13\*          | -              |
| [Mapillary+Cityscapes_2](https://zenodo.org/records/14766160/files/vltseg_checkpoint_mapillary+cityscapes_2.pth?download=1) | 1K + 20K + 40K     | 8          | 0.15             | -         | -          | -            | 85.50           | 86.39\*          | -              |
| [Mapillary_1](https://zenodo.org/records/14766160/files/vltseg_checkpoint_mapillary_1.pth?download=1)                       | 20K                | 8          | 0.20             | 74.76     | -          | 68.99        | 81.51           | -                | 84.01          |

_[\*] All submissions to the ACDC and Cityscapes test set benchmarks used test time augmentations, as described in Section 6.2 of the paper._

When a checkpoint lists "1K + \_\_K" iterations, it was trained for 1K iterations with a frozen backbone before the regular training. The checkpoints Mapillary+Cityscapes\_[1/2] were trained for 20K iterations on Mapillary and 40K iterations on Cityscapes, as described in Section 6.2 of the paper.

## License

The code in this repository is licensed under the [MIT License](https://opensource.org/license/mit).

The checkpoints are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/).
