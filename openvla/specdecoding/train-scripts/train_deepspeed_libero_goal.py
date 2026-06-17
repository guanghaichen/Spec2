import deepspeed
import os
import argparse
import wandb
import sys
import time
from datetime import datetime
sys.path.insert(0, "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main")

def _log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[train_deepspeed_libero_goal.py {now}] {msg}", flush=True)

_t0 = time.time()
_log("script start")
parser = argparse.ArgumentParser(description='My training script.')
parser.add_argument('--local_rank', type=int, default=1,
                    help='DeepSpeed 分布式启动器传入的本地 GPU 编号')
#parser.add_argument("--deepspeed_config", type=str, default='/mnt/public/wangsongsheng/home/Projects/20250223-OpenVLA/openvla/specdecoding/scripts/llama_2_chat_7B_config.json',help="accellerate config path")
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
cmd_args = parser.parse_args()
_log(f"parsed args: local_rank={getattr(cmd_args, 'local_rank', None)}")
#os.chdir("/mnt/public/wangsongsheng/home/Projects/20250223-OpenVLA")
basepath="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"# 模型路径
cpdir="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/ckpt_libero_goal_debug_ckpt"# 权重保存目录
tmpdir="/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/specvla-data/libero_goal_dataset"# 训练数据
train_config = {
    "lr": 5e-5,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "datapath": f"{tmpdir}",
    "is_warmup": True,# 是否使用学习率预热
    "num_epochs": 200,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,# 预测损失（ploss）的权重
    "v_w": 1.0,# 值损失（vloss）的权重
    "head_w": 0.1,# head 损失权重（未使用）
    "num_workers": 1,
    "embeding": True,
    "act": "No",
    "data_noise": True,# 是否对数据加噪声（数据增强）
    "noise": "uniform",# 噪声类型：均匀分布
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    "config_path": "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/SpecVLA-main/openvla/specdecoding/train-scripts/llama_2_chat_7B_config.json",# 小模型Draft配置路径
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}
from safetensors import safe_open
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
# 允许在 Ampere 架构（A100等）GPU 上使用 TF32，加速矩阵乘法
torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
_log("creating Accelerator(fp16)")
accelerator = Accelerator(mixed_precision="fp16")
_log(f"Accelerator created in {time.time() - _t0:.2f}s")
from openvla.specdecoding.model.cnets import MMModel
#from configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
# 设置分布式环境变量（DeepSpeed 需要）
os.environ['MASTER_ADDR']='localhost'
os.environ['MASTER_PORT']='14756'
os.environ['WANDB_MODE']='offline'# wandb 离线模式（不联网上传）
_log("calling deepspeed.init_distributed()")
deepspeed.init_distributed()
rank = torch.distributed.get_rank()
_log(f"distributed initialized: rank={rank}, world_size={torch.distributed.get_world_size()}")
if rank == 0:
    import wandb

    wandb.init(project="OpenVLA", entity="ss_wang", config=train_config)

from typing import Optional, Union
from pathlib import Path
'''from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)'''
from pathlib import Path
from typing import Optional
from transformers import AutoModelForVision2Seq
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/mnt/3b51049a-abd1-486a-89ce-cfd16ced42a8/cgh/data/models--openvla--openvla-7b-finetuned-libero-goal"                            # Path to OpenVLA model (on HuggingFace Hub)
# cfg=parser.parse_args()
cfg=FinetuneConfig()
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
_log(f"loading vla from_pretrained: {cfg.vla_path}")
_t_model = time.time()
# 加载预训练模型
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.vla_path,
    torch_dtype=torch.bfloat16,
    quantization_config=None,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
_log(f"vla loaded in {time.time() - _t_model:.2f}s; total {time.time() - _t0:.2f}s")

# 提取语言模型部分（这里只用它的 lm_head 来做“token 分布蒸馏”的监督；lm_head 参数会被冻结）
lm = vla.language_model
# 语言模型的输出头lm head
vla_lm_head = lm.lm_head
# 获取 lm_head 的维度信息
vocab_size,hidden_dim=vla_lm_head.out_features,vla_lm_head.in_features
# 取出lm head的权重矩阵
tensor = vla_lm_head.weight.data
# 创建一个新的 Linear 层，复制大模型的 lm_head 权重
head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor
_log(f"head prepared: vocab_size={vocab_size}, hidden_dim={hidden_dim}")

def list_files(path):
    """递归遍历目录，返回所有文件路径"""
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class AddGaussianNoise:
    """添加高斯噪声到 hidden_state_big 中"""
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    """添加均匀分布噪声到 hidden_state_big 中"""
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    """
    自定义数据集：读取 ge_data_all_openvla_token_only_libero_goal.py 生成的单样本 .ckpt。

    每个 .ckpt 里核心字段：
    - predicted_tokens: (7,) 动作 token（OpenVLA 一次生成 7 个 action tokens）
    - hidden_state: (first_layer_hidden_list, last_layer_hidden_list)
      * first_layer_hidden_list: 长度 T 的 list，每个元素形状通常是 (1, hidden_dim)
      * last_layer_hidden_list : 长度 T 的 list，每个元素形状通常是 (1, hidden_dim)

    本脚本训练 Draft（MMModel / EA layer）的目标是：
    - 输入：teacher 的 last-layer hidden 序列 + “embedding/浅层表征”序列
    - 输出：预测的 next-step hidden（用于后续 topK/tree draft）
    - 监督：hidden 回归（SmoothL1） + token 分布蒸馏（冻结 lm_head）
    """
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform# 数据增强

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        # 加载单个数据文件
        data = torch.load(self.data[index])
        new_data = {}
        # teacher 的“最后层 hidden state”序列（拼成 [T, H]）（同一层的7个hidden拼在一起）
        # 这里的 T 通常是动作自回归生成步数（常见为 7）
        hidden_state = torch.cat([item for item in data['hidden_state'][1]],dim=0)
        # teacher 的“第一层/embedding 输出”序列（拼成 [T, H]）
        # 训练时作为 Draft 的 input_embeddings 输入（帮助 Draft 感知 token embedding/浅层表征）
        embedding_state = torch.cat([item for item in data['hidden_state'][0]],dim=0)
        # 本样本动作 token（长度 7）；训练时用于构造监督 mask，并可用于额外对齐/调试
        target_tokens = torch.tensor(data['predicted_tokens'])
        input_ids = data['input_ids']
        loss_mask = data["loss_mask"]
        pixel_values = data["pixel_values"]
        # length 这里用于构造“前缀长度”，但数据中 hidden_state 的每步张量第一维通常是 1，
        # 因此 length 多数情况下为 0。实际训练监督是通过下面的 loss_mask 强制只监督 7 个动作 token 段。
        length = data['hidden_state'][0][0].shape[0]-1
        attention_mask = [1] * (length)
        # loss_mask：只对动作 token 段计算损失（长度 7）
        # 注意：最后一个位置会被置 0，用于对齐/避免越界（见下方 target/embedding padding）
        loss_mask = [0]*(length) + [1]*7
        loss_mask[-1] = 0
        # input_ids_target：训练时传给 Draft 的 input_ids（主要用于位置长度/对齐）
        # 这里把动作 token 拼到末尾；非动作部分用 0 padding
        input_ids_target = torch.cat([torch.tensor([0]*(length-1)),target_tokens,torch.tensor([0])])
        # target：teacher 最后层 hidden 的“下一步”监督（next-step hidden prediction）
        # 即用 hidden_state[t+1] 监督 predict[t]（随后会做 padding 对齐到相同长度）
        target = hidden_state[1:, :]
        # embedding_state 也做同样的 shift，保证与 target/predict 的时间步对齐
        embedding_state = embedding_state[1:, :]
        zeropadding = torch.zeros(1, target.shape[1])
        # 在序列尾部补 1 个 0 向量，使 target/embedding_state 与 hidden_state_big 的长度一致
        target = torch.cat((target, zeropadding), dim=0)
        embedding_state = torch.cat((embedding_state, zeropadding), dim=0)
        loss_mask[-1] = 0
        # 输出字典字段含义（单样本）：
        # - hidden_state_big: [T, H] teacher 最后层 hidden 序列（Draft 的主输入之一）
        # - embedding_state:  [T, H] teacher 第一层/embedding 序列（Draft 的 input_embeddings）
        # - target:           [T, H] teacher 下一步最后层 hidden（监督）
        # - input_ids:        [T]   主要用于对齐/位置长度（动作 token 段在末尾）
        # - attention_mask:   [T]   传给 Draft 的 attention mask
        # - loss_mask:        [T]   只对动作 token 段监督（其余为 0）
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["embedding_state"] = embedding_state
        new_data["input_ids"] = input_ids_target
        new_data['pixel_values'] = pixel_values


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:
    """
    把变长序列 padding 成 batch。

    输出 batch 的关键字段（形状以 batch_size=B、最大长度=max_length、hidden_dim=H 表示）：
    - input_ids:         [B, max_length]
    - hidden_states:     [B, max_length, H]   (teacher last-layer hidden 序列)
    - embedding_states:  [B, max_length, H]   (teacher first-layer/embedding 序列)
    - target:            [B, max_length, H]   (teacher next-step last-layer hidden)
    - attention_mask:    [B, max_length]
    - loss_mask:         [B, max_length]      (只监督动作 token 段)
    - pixel_values:      [B, ...]             (本训练脚本未使用 pixel_values 做 forward，仅随 batch 保留)
    """

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        n, d = intensors.shape
        padding_tensor = torch.zeros(N - n, d, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=0)
        return outtensors
    def paddingtensor1D(self, intensors, N):
        n = intensors.shape[0]
        if N>n:
            padding_tensor = torch.zeros(N - n, dtype=intensors.dtype)
            outtensors = torch.cat((intensors, padding_tensor), dim=0)
            return outtensors
        elif N < n:
            print('error!!!',N,n)
            return intensors
        else:
            return intensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max([item['hidden_state_big'].shape[0] for item in features])
        batch_input_ids = torch.cat([self.paddingtensor1D(item['input_ids'],max_length).unsqueeze(0) for item in features],dim=0)
        batch_hidden_states = torch.cat([self.paddingtensor2D(item['hidden_state_big'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_embedding_states = torch.cat([self.paddingtensor2D(item['embedding_state'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_target = torch.cat([self.paddingtensor2D(item['target'], max_length).unsqueeze(0) for item in features],dim=0)
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_pixel_values = torch.cat([item['pixel_values'] for item in features])
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "embedding_states":batch_embedding_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            'pixel_values':batch_pixel_values.to(torch.bfloat16)
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

def compute_loss(target, target_p, predict, loss_mask):
    """
    计算两类损失：
    1) vloss：Draft 预测 hidden 与 teacher next-step hidden 的回归损失（SmoothL1），只在 loss_mask=1 的位置计算。
    2) ploss：token 分布蒸馏损失。把 Draft 的 predict 与 teacher 的 target 都过冻结 lm_head，
       用 teacher 的 soft target_p 对 Draft 的 log-prob 做交叉熵式蒸馏，同样受 loss_mask 约束。

    输入：
    - target:   [B, T, H] teacher next-step last-layer hidden
    - target_p: [B, T, V] teacher token 分布（由冻结 lm_head(target) 得到 softmax）
    - predict:  [B, T, H] draft 输出 hidden
    - loss_mask:[B, T, 1] 只监督动作 token 段

    输出：
    - vloss: 标量
    - ploss: 标量
    - out_head: [B, T, V] Draft 的 vocab logits（便于统计 topk accuracy）
    """
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1])
    return vloss, ploss, out_head

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

_log(f"scanning datapath: {train_config['datapath']}")
_t_scan = time.time()
datapath = list_files(train_config["datapath"])
_log(f"scan done: files={len(datapath)} in {time.time() - _t_scan:.2f}s")

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
_log(f"split: train={len(traindatapath)}, test={len(testdatapath)}")
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
_log("creating test DataLoader")
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

from openvla.specdecoding.model.configs import EConfig
from openvla.specdecoding.model.cnets import MMModel

if rank == 0:
    if not os.path.exists(cpdir):
        os.makedirs(cpdir)

_log(f"loading EConfig: {train_config['config_path']}")
config = EConfig.from_pretrained(train_config["config_path"])

_log("creating MMModel")
model = MMModel(config, path=basepath, load_emb=True)

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]
_log("calling deepspeed.initialize for main model")
_t_ds = time.time()
model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=cmd_args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )
_log(f"deepspeed.initialize(main) done in {time.time() - _t_ds:.2f}s")

_log("calling deepspeed.initialize for head")
_t_ds_head = time.time()
head_engine, _, test_loader, _ = deepspeed.initialize(args=cmd_args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )
_log(f"deepspeed.initialize(head) done in {time.time() - _t_ds_head:.2f}s")
for param in head.parameters():
    param.requires_grad = False
print('start training')
global_step = 0
for epoch in range(num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    pbar = train_loader
    if rank == 0:
        pbar = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True, desc=f"train {epoch + 1}/{num_epochs}")
    for batch_idx, data in enumerate(pbar):
        model.zero_grad()
        # Draft forward：
        # - data["hidden_states"]      : teacher last-layer hidden 序列 [B, T, H]
        # - data["embedding_states"]   : teacher first-layer/embedding 序列 [B, T, H]
        # - data["input_ids"]          : 对齐用 token 序列 [B, T]
        # - data["attention_mask"]     : [B, T]
        predict = model_engine(data["hidden_states"].to(rank), input_ids=data["input_ids"].to(rank),input_embeddings=data['embedding_states'].to(rank),
                               attention_mask=data["attention_mask"].to(rank))
        with torch.no_grad():
            # teacher 监督：把 target hidden 过冻结 lm_head 得到 teacher token 分布
            target_head = head_engine(data["target"].to(rank))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()
        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
        model_engine.backward(loss)

        model_engine.step()
        global_step += 1

        with torch.no_grad():
            # 评估：只在 loss_mask=1 的位置统计 token 分类准确率（通过冻结 lm_head 输出 vocab logits）
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    step=global_step,
                    loss=f"{loss.item():.4f}",
                    v=f"{vloss.item():.4f}",
                    p=f"{ploss.item():.4f}",
                    acc=f"{(cc / ct):.3f}",
                    lr=f"{optimizer.optimizer.param_groups[0]['lr']:.2e}",
                )

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

    model_engine.save_16bit_model(f"{cpdir}/state_{epoch}")
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{cpdir}/state_{epoch}")
        print('checkpoint saved')
