## Setup Guidance
#### Requirements & Installation

* Python >= 3.10
* Pytorch == 2.2.0 (tested with cuda == 12.1)
* Libero == 0.1.0
* ``pip install -r requirements-min.txt``
* ``cd openvla``
* ``pip install -e .``

## Main Files

We derive our resposary from OpenVLA, and provide the impelementation for the SpecVLA framework. The main architecture is listed below: 

```
SpecVLA
├── openvla
│   ├── experiments                # Scripts for conducting libero simulation benchmark and speedup test
│   ├── prismatic                  # Derived from the openvla
│   ├── scripts                    # Derived from the openvla
|   └── specdecoding               # SpecVLA implementation
├── dataset                        # Finetuning dataset 
└── backbone_models                # Finetuned OpenVLA models
```
### Draft Model Checkpoints (LIBERO)

We provide pretrained draft model checkpoints for all four LIBERO task suites.  
Each checkpoint can be directly used for Spec-VLA speculative decoding without additional training.

<div align="center">

| LIBERO Task Suite | Draft Model Checkpoint |
|-------------------|------------------------|
| **LIBERO Goal**   | [Download](https://drive.google.com/drive/folders/1W7nBHM9-bf9tq4NQDUfles583OUfrtLv?usp=share_link) |
| **LIBERO Object** | [Download](https://drive.google.com/drive/folders/1HHQv5iRMXRSfajjIgB62h_jMGboXu250?usp=share_link) |
| **LIBERO Spatial**| [Download](https://drive.google.com/drive/folders/1Het7jUEiWSObG8Tn7H2CjmqdH_XdJr5x?usp=share_link) |
| **LIBERO 10**     | [Download](https://drive.google.com/drive/folders/1LhV2bAzdivbaz6MM1Owl_jNCCdnNs-8W?usp=share_link) |

</div>



#### Experiment Pipeline

#### Training data generation
```
python SpecVLA/openvla/specdecoding/train-scripts/ge_data_all_openvla_token_only_libero_goal.py
```
#### Training Draft models
```
export PYTHONPATH='/SpecVLA-main'
WANDB_MODE='offline' deepspeed --master_port 23333 --include=localhost:0 "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/train-scripts/train_deepspeed_libero_goal.py" --deepspeed_config "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/train-scripts/ds_config.json"
```
#### Testing on LIBERO simulation benchmark
Autoregressive Generation
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/experiments/robot/libero/run_libero_goal_AR.py\
  --model_family openvla \
  --pretrained_checkpoint /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```
Speculative Decoding
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/experiments/robot/libero/run_libero_goal_Spec.py \
    --model_family openvla \
    --pretrained_checkpoint /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True
```
Speculative Decoding with Relaxed Acceptance
```
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed.py \
    --model_family openvla \
    --pretrained_checkpoint /mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True
```

