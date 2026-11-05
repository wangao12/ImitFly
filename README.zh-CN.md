# ImitFly 中文说明

本仓库对应论文 **ImitFly: Bio-Inspired Sensorimotor Priors Enable Sample-Efficient Cross-Embodiment Control**，
以原项目中的 `model_meansnn51` 系列为果蝇飞行主模型，按论文术语组织为：

- **SCVE**：脉冲卷积视觉编码器，提取双目视觉特征。
- **LSTM**：分别编码视觉历史与传感器序列。
- **HMAM**：传感器注意力与特征组融合注意力。
- **Gaussian policy / KL distillation**：输出动作分布，通过专家分布进行离线蒸馏。

仓库提供果蝇飞行、果蝇步行适配、对比模型和消融配置。无人机单目导航部分不包含在此目录中。
权重、专家策略与数据集另行提供，不包含在此代码目录中。

## 安装

建议 Python 3.10，在本目录中执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[sim]'
```

也可使用 `requirements.txt` 中的参考版本。`environment.yml` 用于 Conda 安装。
学生模型的训练和推理不依赖 TensorFlow；专家数据采集使用单独环境及 `requirements-teacher.txt`。

## 训练

将 12 条示范保存为 `data/flight/episode_0001.npz` 至 `episode_0012.npz`：

```bash
python -m imitfly.cli.train \
  --config configs/imitfly_12_demos.json \
  --data data/flight \
  --output outputs/imitfly_12_demos
```

示范编号、训练轮数、批量大小、学习率和验证划分均在 JSON 配置中修改。
`imitfly_12_demos.json` 对应论文的 12 条示范设置；
`imitfly_v51.json` 保留当前原始 v51 脚本的 18 条示范、260 轮设置，便于继续使用现有实验。
训练输出包括权重、优化器状态、上一份检查点、配置和损失曲线数据。

## 推理与视频

将单独提供的 v51 权重放入 `checkpoints/`，并提供 FlyBody 基础翼拍模式文件：

```bash
export MUJOCO_GL=egl
python -m imitfly.cli.evaluate \
  --checkpoint checkpoints/imitfly_v51.pth \
  --wpg data/assets/wing_pattern_fmech.npy \
  --output outputs/flight_demo \
  --seeds 0 1 2 \
  --video
```

仿真按 0.2 ms 控制间隔运行，正常时长为 400 ms。视频以 30 fps 播放，为慢动作展示。
轨迹文件包含仿真时间、位置、速度和动作，便于后续绘图。

## 其他入口

- 采集专家数据：`python -m imitfly.cli.collect`。
- 迁移旧日志：`python -m imitfly.cli.convert_data`。
- 步行训练：使用 `configs/walking.json` 和同一个训练入口。
- 步行推理：`python -m imitfly.cli.evaluate_walking`。
- 结构消融与传感器消融：见 [实验说明](docs/EXPERIMENTS.md)。
- 数据字段与输入顺序：见 [数据格式](docs/DATA_FORMAT.md)。
- 原文件与新文件对应关系：见 [源码映射](docs/SOURCE_MAP.md)。

主模型按 v51 的实际层结构整理，保持原始状态字典的主要参数名称。
新提供的消融开关用于统一组织相应实验，需要分别训练并使用各自的权重。
本次整理没有重新复现论文实验；论文结果应使用相应的正式数据和模型权重。
