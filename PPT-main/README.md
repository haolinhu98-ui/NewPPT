<p align="center">
<strong>Progressive Pretext Task Learning for Human Trajectory Prediction</strong></h1>
  <p align="center">
    <a href='https://xiaotong-lin.github.io/' target='_blank'>Xiaotong Lin</a>&emsp;
    <a href='https://tmliang.github.io/' target='_blank'>Tianming Liang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=w3GjGqoAAAAJ' target='_blank'>Jianhuang Lai</a>&emsp;
    <a href='https://www.isee-ai.cn/~hujianfang/' target='_blank'>Jian-Fang Hu*</a>&emsp;
    <br>
    Sun Yat-sen University
    <br>
    ECCV 2024
  </p>
</p>

</p>
<p align="center">
  <a href='https://arxiv.org/pdf/2407.11588'>
    <img src='https://img.shields.io/badge/Arxiv-2407.11588-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04345.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>

  <a href='https://github.com/iSEE-Laboratory/PPT'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>

  </a>
</p>

## 🏠 Abstract
<div style="text-align: center;">
    <img src="assets/Intro_cmp.jpg" width=100% >
</div>
Human trajectory prediction is a practical task of predicting the future positions of pedestrians on the road, which typically covers all temporal ranges from short-term to long-term within a trajectory. However, existing works attempt to address the entire trajectory prediction with a singular, uniform training paradigm, neglecting the distinction between short-term and long-term dynamics in human trajectories. To overcome this limitation, we introduce a novel Progressive Pretext Task learning (PPT) framework, which progressively enhances the model's capacity of capturing short-term dynamics and long-term dependencies for the final entire trajectory prediction. Specifically, we elaborately design three stages of training tasks in the PPT framework. In the first stage, the model learns to comprehend the short-term dynamics through a stepwise next-position prediction task. 
In the second stage, the model is further enhanced to understand long-term dependencies through a destination prediction task. 
In the final stage, the model aims to address the entire future trajectory task by taking full advantage of the knowledge from previous stages. To alleviate the knowledge forgetting, we further apply a cross-task knowledge distillation. Additionally, we design a Transformer-based trajectory predictor, which is able to achieve highly efficient two-step reasoning by integrating a destination-driven prediction strategy and a group of learnable prompt embeddings. Extensive experiments on popular benchmarks have demonstrated that our proposed approach achieves state-of-the-art performance with high efficiency.
</br>

## 📖 Implementation
### I. Installation
#### Environment
 - Python == 3.8.3
 - PyTorch == 1.7.0

#### Dependencies

Install the dependencies from the `requirements.txt`:
```linux
pip install -r requirements.txt
```

#### Pre-trained Models and Pre-processed Data

We provide a complete set of pre-trained models, including:

* Well-pretrained model on Task-I:
* The model after warm-up:
* Well-pretrained model on Task-II:
* Well-trained model on Task-III:

You can download the pre-trained models and the pre-processed data from [here](https://drive.google.com/drive/folders/13ncPnZPFE7cPHZ8KiCYkfeC_2avbg_lr?usp=sharing).

#### File Structure

After the preparation work, the whole project should has the following structure:

```
./MemoNet
├── README.md
├── data                            # datasets
│   ├── ETH_UCY
│   │   ├── social_eth_test_256_0_50.pickle
│   │   ├── social_eth_train_256_0_50.pickle
│   │   └── ...
│   ├── ETH_image
│   │   ├── eth.jpg
│   │   ├── eth_H.txt
│   │   └── ...
│   ├── social_sdd_test_4096_0_100.pickle
│   └── social_sdd_train_512_0_100.pickle
│   ├── jaad
│   │   ├── train.pkl
│   │   └── test.pkl
│   └── pie
│       ├── train.pkl
│       └── test.pkl
├── models                          # core models
│   ├── layer_utils.py
│   ├── model.py
│   └── ...
├── requirements.txt
├── run.sh
├── sddloader.py                    # sdd dataloader
├── test_PPT.py                     # testing code
├── train_PPT.py                    # training code
├── trainer                         # core operations to train the model
│   ├── evaluations.py
│   ├── test_final_trajectory.py
│   └── trainer_AIO.py
└── training                        # saved models/memory banks
    └── Pretrained_Models
       ├── SDD
       │    ├── Model_ST
       │    ├── Model_Des_warm
       │    ├── Model_LT
       │    └── Model_ALL
       └── ETH_UCY
           ├── model_eth_res
           ├── model_hotel_res
           └── ...
    
```

#### JAAD/PIE data format
To use the JAAD/PIE datasets, prepare `train.pkl` and `test.pkl` (or `.npz`) under
`data/jaad` and `data/pie`. Each file should contain one of the following payloads:

* **Dict payload** with keys:
  * `trajectories`: array shaped `[num_samples, num_agents, seq_len, 2]`
  * `masks`: array shaped `[num_samples, num_agents, num_agents]` describing scene membership
  * `seq_start_end` (optional): list of `(start, end)` pairs per sample
  * `initial_pos` (optional): array shaped `[num_samples, num_agents, 2]`
  * `maps` (optional): array shaped `[num_samples, C, H, W]` for semantic raster maps
* **Tuple/List payload**:
  * `(trajectories, masks)` or `(trajectories, masks, seq_start_end[, initial_pos[, maps]])`

If `seq_start_end` or `initial_pos` are not provided, they are derived automatically in the loader.
Set `--dataset_name jaad` or `--dataset_name pie` and optionally override `--data_root` if your
data is stored elsewhere.

#### JAAD/PIE preprocessing template
We provide a template script to convert per-frame annotations into the expected payload:

```linux
python tools/prepare_jaad_pie.py \
  --input_csv /path/to/jaad_or_pie.csv \
  --output_dir data/jaad \
  --split train \
  --obs_len 8 \
  --pred_len 12 \
  --stride 1
```

The CSV must include `scene_id`, `frame_id`, `track_id`, `x`, and `y` columns
(rename via `--scene_col`, `--frame_col`, `--track_col`, `--x_col`, `--y_col`).

#### JAAD XML to CSV helper (with image-size normalization)
If you start from JAAD `annotations/*.xml`, use the helper below to produce the
normalized CSV required by `prepare_jaad_pie.py`. The script reads `original_size`
from each XML (or you can override via `--width/--height`).

```linux
python tools/jaad_xml_to_csv.py \
  /path/to/JAAD/annotations \
  --output_csv /path/to/jaad.csv
```

This generates a CSV with `scene_id`, `frame_id`, `track_id`, `x`, `y` columns
where `x` and `y` are bbox centers normalized by image width/height.

#### JAAD/PIE standard input format & normalization strategy
**Standard input format (recommended):**

* **Coordinates**: use the pedestrian center in image space `(x, y)` for each frame.
* **Sequence length**: `seq_len = obs_len + pred_len` (default 8 + 12 = 20).
* **Payload**: follow the dict/tuple formats above (`trajectories`, `masks`, optional
  `seq_start_end` and `initial_pos`). The template script outputs this structure.

**Normalization strategy (choose one and stay consistent):**

1. **Image-size normalization (recommended for JAAD/PIE):**
   * `x_norm = x / image_width`, `y_norm = y / image_height`
   * Store normalized coordinates directly in `trajectories`.
   * Use the same normalization during evaluation/visualization.
2. **Fixed-scale normalization (if you do not have frame size per sample):**
   * Use a constant scale factor (e.g., `scale=1/1000` or `1/1920`).
   * Keep the same scale for train/test/val and report the scale in experiments.
3. **Per-sequence centering (optional augmentation):**
   * Subtract the last observed position from every timestep inside each sequence:
     `traj_centered = traj - traj[:, obs_len - 1:obs_len, :]`.
   * This is already applied inside the model pipeline; do not double-center in preprocessing.

**Notes:**
* For JAAD/PIE, keep coordinates in the image plane unless you have calibrated
  camera parameters to convert to world coordinates.
* If you add vehicles later, use the same normalization policy for all agent types.

#### Semantic raster map conditioning (optional)
To enable semantic map conditioning, include a `maps` array in the JAAD/PIE payload
with shape `[num_samples, C, H, W]` and run training/testing with:

```linux
python train_PPT.py --dataset_name jaad --use_semantic_map --map_channels C
```

The model uses a lightweight CNN encoder with global pooling to fuse map context
into the trajectory encoder. Keep semantic maps normalized to `[0, 1]`.

### II. Training

Important configurations.

* `--mode`: verify the current training mode, 
* `--model_Pretrain`: pretrained model path,
* `--info`: path name to store the models,
* `--gpu`: number of devices to run the codes,

Training commands.

```linux
bash run.sh
```


### III. Reproduce

To get the reported results, following

```linux
python test_PPT.py --reproduce --info reproduce --gpu 0
```

And the code will output: 

```linux
./training/Pretrained_Models/SDD/model_ALL
Loaded data!
Test FDE_48s: 10.650254249572754 ------ Test ADE: 7.032739639282227
----------------------------------------------------------------------------------------------------
```



### IV. Visualization

We also provide the visualization code for the ETH/UCY dataset. For example, to visualize trajectories in the univ scene, use the following command:

```linux
python test_PPT.py --vis --dataset_name eth --data_scene 'univ' --model_Pretrain './training/Pretrained_Models/ETH_UCY/model_univ' --gpu 0
```

## 🔍 Overview

<p align="center">
  <img src="./assets/Architecture.jpg" width=100% >
</p>
As shown, we propose a Progressive Pretext Task learning (PPT) framework for trajectory prediction, aiming to incrementally enhance the model's capacity to understand the past trajectory and predict the future trajectory.
Specifically, our framework consists of three stages of progressive training tasks, as illustrated in subfigure (b). In Stage I, we pretrain our predictor on pretext Task-I, aiming to fully understand the short-term dynamics of each trajectory, by predicting the next position of a trajectory of arbitrary length. In Stage II, we further train the predictor on pretext Task-II, intending to capture the long-term dependencies, by predicting the destination of a trajectory.
Once Task-I and Task-II are completed, the model is capable of capturing both the short-term dynamics and long-term dependencies within the trajectory. Finally, in Stage III, we duplicate our model to obtain two predictors: one for destination prediction and another for intermediate prediction. In this stage, we perform Task-III that enables the model to achieve the complete pedestrian trajectory prediction.
For the sake of stable training, we further employ a cross-task knowledge distillation to avoid knowledge forgetting.
<!-- 
### 🧪 Experimental Results

#### Qualitative Comparisons with Pure Diffusion
<p align="center">
  <img src="assets/results.png" align="center" width="100%">
</p> -->



## 👏 Acknowledgements

We sincerely thank the authors of [MemoNet](https://github.com/MediaBrain-SJTU/MemoNet?tab=readme-ov-file) for providing the source code from their CVPR 2022 publication. We also appreciate the pre-processed data from [PECNet](https://karttikeya.github.io/publication/htf/). These resources have been invaluable to our work, and we are immensely grateful for their support.



## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
   lin2024progressive,
   title={Progressive Pretext Task Learning for Human Trajectory Prediction},
   author={Lin, Xiaotong and Liang, Tianming and Lai, Jianhuang and Hu, Jian-Fang},
   booktitle={ECCV},
   year={2024},
}
```
