# CelebA Identity Defense/Attack Pipeline

代码只负责准备脚本和命令入口，不会自动提交 Slurm 任务。重任务请放进 `job.sh/run.sh` 后在计算节点执行。

## 数据划分

CelebA 官方划分文件已经存在：`data/celeba/list_eval_partition.csv`。

```bash
python run_pipeline.py split --out work/celeba_official_split.csv
```

身份识别还需要 CelebA 身份标签。请把官方 `identity_CelebA.txt` 或等价 CSV 放到 `data/celeba/`。没有身份标签时，Stage 1/3/4/5 会明确报错。

## Stage 0: 多噪声 buffalo_l 向量缓存

为每张图生成 0% 到 90% 十档高斯噪声 embedding，并用 insightface `buffalo_l` 提取向量。缓存按噪声档位和 shard 保存在 `work/embeddings/noise_XX/`，已存在的完整 shard 默认跳过。

默认使用 `--embedding-mode landmarks`：直接读取 CelebA 官方 5 点 landmarks，跳过 detector，在 112x112 对齐人脸 crop 上加十档噪声，并把很多张图的 crop 聚成大 batch 送入 buffalo_l recognition 模型。这个比对十张整图分别 `app.get()` 快得多。

可选模式：

- `landmarks`：最快，使用 CelebA 官方 landmarks，不跑 detector。
- `fast`：原图只检测/对齐一次，然后十档 crop 批量识别。
- `full`：最原始也最慢，十张整图分别 `app.get()`。

```bash
python run_pipeline.py stage0 \
  --batch-size 1024 \
  --ctx-id 0 \
  --save-every 512 \
  --embedding-mode landmarks \
  --recognition-batch-size 1024 \
  --preprocess-workers 8
```

如果要缓存 SDXL VAE latent，请单独手动跑下面的命令。主 `run.sh` 不再自动碰 VAE cache，避免已经完成的 latent 在 embedding 作业里被反复检查或重新处理：

```bash
python run_pipeline.py stage0 \
  --skip-embeddings \
  --cache-vae \
  --vae-batch-size 8 \
  --vae-image-size 512 \
  --device cuda
```

Stage 0 支持进度条；如果没有安装 `tqdm`，会退化成普通进度日志。embedding shard 在分片内也会 partial 落盘，重启后会从最短公共进度继续跑；完整 shard 会带 `complete=True` 并自动跳过。

如果日志里 buffalo_l 显示 `Applied providers: ['CPUExecutionProvider']`，说明 ONNXRuntime 没有用上 GPU，必须停止作业并安装/启用 `onnxruntime-gpu` 后重跑。`run.sh` 已经会检查 `CUDAExecutionProvider`，并在不可用时安装 `onnxruntime-gpu==1.20.1`；代码也会拒绝在 GPU 作业中静默 fallback 到 CPU。

默认作业只申请 1 张 GPU；当前只保留 `job.sh -> run.sh` 这一条入口。

## Stage 1: 原系统检索

CelebA 官方划分的 train/val/test 身份不重叠，所以默认用测试集内部检索：测试集图片做 query，测试集图片做 gallery，并自动排除自身匹配，报告 top1/top5。

```bash
python run_pipeline.py stage1 --query-split test --gallery-split test --level 0
```

指标输出到 `work/metrics/stage1_retrieval.json`。

## Stage 2: 攻击原系统

先把 0% 噪声 embedding 合并成一个文件：

```bash
python run_pipeline.py export-level --level 0
```

生成 512x512 重建图：

```bash
python run_pipeline.py stage2 \
  --embedding-file work/embeddings/buffalo_l_noise_00.pt \
  --generate
```

评估 InsightFace cosine 和 CLIP ViT-L/14 cosine：

```bash
python run_pipeline.py stage2 \
  --embedding-file work/embeddings/buffalo_l_noise_00.pt \
  --generated-dir work/generated/stage2 \
  --evaluate
```

## Stage 3: 防御模型 ArcFace 训练

模型结构：十档 embedding 的 softmax 加权和，再接 `512->384->512` MLP（若 buffalo_l 输出维度不是 512，则 hidden 自动取 3/4）。

```bash
python run_pipeline.py stage3 --epochs 20 --batch-size 512 --lr 1e-3
```

checkpoint 默认保存到 `work/checkpoints/defense_arcface.pt`。

导出新模型 embedding：

```bash
python run_pipeline.py export-defended --ckpt work/checkpoints/defense_arcface.pt --split train
python run_pipeline.py export-defended --ckpt work/checkpoints/defense_arcface.pt --split test
```

## Stage 4: 防御后攻击 A

直接用新模型输出向量重建并评估：

```bash
python run_pipeline.py stage4 \
  --embedding-file work/embeddings/defended_test.pt \
  --generate --evaluate
```

## Stage 5: 防御后攻击 B

先用训练集的新 embedding 微调 IP-Adapter/UNet LoRA：

```bash
python run_pipeline.py stage5 \
  --embedding-file work/embeddings/defended_train.pt \
  --steps 1000 --batch-size 1 --lr 1e-4
```

Stage 5 依赖 `ip_adapter` 包的 `IPAdapterFaceIDXL` 运行时 API。不同版本 API 可能不一致，如果报错，优先检查已安装的 IP-Adapter 代码版本和 `get_image_embeds`/attention processor 支持。

## 依赖提示

计算节点环境至少需要：

```bash
pip install opencv-python insightface onnxruntime-gpu diffusers transformers accelerate peft safetensors
```
