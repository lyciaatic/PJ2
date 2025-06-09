# 神经网络与深度学习 - Project2报告

**刘滢 21300110095**  
**GitHub代码链接：** (https://github.com/lyciaatic/PJ2) 
**数据集及模型权重下载链接：** https://drive.google.com/drive/folders/1Tx5LAh7sQ5A-OuKI97b9EbYsguWlr7bq

---

## 一、项目概述
本项目总体分为两部分，旨在基于CIFAR-10图像分类数据集，通过构建不同结构的卷积神经网络（CNN）模型，尝试研究Batch Normalization（BN）等深度学习技术在训练优化过程中的作用与影响。主要任务包括模型训练、性能对比分析、优化实验、梯度与特征可视化等。

---

## 二、模型训练与性能分析

### 2.1 数据集
- **数据集名称：** CIFAR-10（10 类，60,000 张 32x32 彩色图片）
- **下载：** torchvision.datasets.CIFAR10
- **数据增强：**
  - RandomCrop、RandomHorizontalFlip、ToTensor、Normalize。
- **归一化均值与标准差：** 均值为 (0.5, 0.5, 0.5)，标准差为 (0.5, 0.5, 0.5)

### 2.2 模型架构
本项目设计并实现了三种 VGG-A 模型变体：
- `VGG_A`：标准 VGG-A 网络
- `VGG_A_BatchNorm`：在标准结构基础上添加了 Batch Normalization 层
- `VGG_A_Residual`：引入残差连接以改善深层网络性能

#### 这些模型均包含：

- 卷积层（Conv2d）

  ```python
   nn.Conv2d(inp_ch, 64, kernel_size=3, padding=1),
   nn.Conv2d(64, 128, kernel_size=3, padding=1),
   # ... 更多卷积层
   ```

- 最大池化层（MaxPooling）
- 激活函数（ReLU）
- 全连接层（Linear）
  
```python
   self.classifier = nn.Sequential(
       nn.Linear(512 * 1 * 1, 512),
       nn.ReLU(True),
       nn.Dropout(0.5),
       nn.Linear(512, 512),
       nn.ReLU(True),
       nn.Dropout(0.5),
       nn.Linear(512, num_classes)
   )
```

- BatchNorm或Residual Connection

```python
   nn.BatchNorm2d(64),
   nn.BatchNorm2d(128),
   # ... 每个卷积层后添加BN
   ```

   ```python
   class ResidualBlock(nn.Module):
       def forward(self, x):
           out = self.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           out += self.shortcut(x)  # 残差连接
           out = self.relu(out)
           return out
   ```

- 此外使用了Xavier初始化卷积层和全连接层，BatchNorm层初始化为常数

### 2.3实现概述

- 实现包含了完整的神经网络架构和所有要求的组件，并添加了多种高级优化策略。
- 具体实现代码细节：
```python
# VGG-A基础架构实现
self.features = nn.Sequential(
    # Stage 1: 3→64通道
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    # Stage 2: 64→128通道
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # ... 后续类似
)

self.classifier = nn.Sequential(
    nn.Linear(512 * 1 * 1, 512),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(512, 512),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(512, 10)  # CIFAR-10的10个类别
)
```

- 尝试实现了批归一化，Dropout，残差连接（自定义残差块，解决深度网络训练），高级数据增强包括Cutout、AutoAugment，标签平滑和焦点损失
    - 残差块实现代码示例：
  
 ```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 通道数不匹配时使用1x1卷积
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out
```

### 2.4 模型训练配置
- **损失函数：** CrossEntropyLoss（交叉熵）
- **优化器：**
  - SGD+Momentum
  - Adam
  - 优化器性能对比结果如下：
  
| 优化器 | 最终验证准确率 | 最佳验证准确率 | 收敛轮数 | 特点 |
|--------|---------------|---------------|----------|------|
| SGD | 89.2% | 89.5% | 15 | 稳定但收敛慢 |
| Adam | 88.8% | 89.1% | 12 | 快速收敛 |
| AdamW | 89.6% | 89.8% | 13 | 更好的泛化 |
| RMSprop | 88.4% | 88.7% | 16 | 稳定但较慢 |
| Custom SGD | 89.4% | 89.7% | 14 | Nesterov改进 |

- **学习率调度器：**
  - CosineAnnealingLR
  - StepLR
- **批量大小**
  - 128
- **权重初始化策略：**
  - 卷积/全连接层采用 Xavier 初始化
  - BN 层采用常数初始化（权重1，偏置0）

### 2.5 架构详述

#### 基础VGG-A架构适配

针对CIFAR-10的32×32输入尺寸，对原始VGG-A进行了适配：

```python
# 适配CIFAR-10的5阶段架构
阶段1: Conv(3→64) → ReLU → MaxPool (32×32 → 16×16)
阶段2: Conv(64→128) → ReLU → MaxPool (16×16 → 8×8)  
阶段3: Conv(128→256) → Conv(256→256) → ReLU → MaxPool (8×8 → 4×4)
阶段4: Conv(256→512) → Conv(512→512) → ReLU → MaxPool (4×4 → 2×2)
阶段5: Conv(512→512) → Conv(512→512) → ReLU → MaxPool (2×2 → 1×1)
分类器: FC(512→512) → FC(512→512) → FC(512→10)
```

#### 实现的模型变体

| 模型名称 | 参数数量 | 架构特点 | 创新点 |
|---------|----------|----------|--------|
| VGG_A | 9,231,114 | 标准VGG无BN | 基础对比模型 |
| VGG_A_BatchNorm | 9,233,418 | VGG + 批归一化 | 每个卷积层后添加BN |
| VGG_A_Residual | 9,512,458 | VGG + 残差连接 + BN | 解决梯度消失问题 |
| VGG_A_Residual_Dropout | 9,512,458 | 残差 + 额外Dropout | 正则化 |
| VGG_A_GELU | 9,233,418 | VGG + BN + GELU激活 | GELU激活函数 |

### 训练结果与性能分析

#### 详细训练过程

我们对每个模型进行了30轮训练，使用以下配置：
- 批大小：128
- 初始学习率：0.01（BN模型为0.02）
- 优化器：SGD with momentum=0.9
- 权重衰减：5e-4
- 学习率调度：在第20和25轮时乘以0.1

**训练过程观察:**

从训练曲线看出：

1. **标准VGG_A**: 
   - 收敛较慢，前10轮准确率提升缓慢
   - 训练曲线有明显波动
   - 最终收敛到86.43%验证准确率

2. **VGG_A_BatchNorm**:
   - 收敛速度明显加快，前5轮就达到80%以上
   - 训练曲线更加平滑
   - 能够使用更高的学习率（0.02 vs 0.01）
   - 最终达到89.54%验证准确率

3. **VGG_A_Residual**:
   - 表现最佳，训练和验证曲线都很稳定
   - 没有明显的过拟合现象
   - 达到最高的91.36%验证准确率

4. **VGG_A_Residual_Dropout**:
   - 由于过度正则化，性能反而下降
   - 训练准确率明显低于验证准确率
   - 说明dropout参数需要仔细调节

5. **VGG_A_GELU**:
   - 训练曲线非常平滑
   - 与ReLU相比收敛更稳定
   - 最终准确率与BatchNorm版本相当

#### 模型性能综合对比

| 模型 | 最终训练准确率 | 最终验证准确率 | 最佳验证准确率 | 训练特征 |
|------|----------------|----------------|----------------|----------|
| VGG_A | 92.00% | 86.43% | 86.43% | 收敛慢，不够稳定 |
| VGG_A_BatchNorm | 95.47% | 89.54% | 89.54% | 收敛快，更稳定 |
| VGG_A_Residual | 97.13% | 91.36% | **91.36%** | 性能最好，训练稳定** |
| VGG_A_Residual_Dropout | 86.08% | 84.49% | 84.49% | 过度正则化 |
| VGG_A_GELU | 96.53% | 89.42% | 89.42% | 训练曲线平滑 |

![训练损失与准确率曲线](./figs/training_loss_and_accuracy.png)



#### 分析
- 从训练损失、训练准确率与验证准确率来看，加入 BatchNorm、残差结构、GELU 激活等方式均能显著加快收敛速度并提升最终精度。特别是 VGG_A_Residual 表现最佳，验证精度高达 91.36%。而使用 Dropout 后虽有一定正则作用，但验证精度略有下降。
此外，尝试 Focal Loss 出现严重欠拟合，说明其可能不适合本任务；而 CutMix 等数据增强策略也有不错的提升效果。

1. **残差连接的重要性**: 
   - 获得最佳性能（91.36%验证准确率）
   - 解决了深度网络的梯度消失问题
   - 允许网络学习更复杂的特征表示

2. **批归一化的显著效果**: 
   - 相比标准VGG提升3-4%的一致改进
   - 训练速度提升约2倍
   - 允许使用更高的学习率

3. **GELU激活函数**: 
   - 与ReLU性能相当但训练更平滑
   - 在某些任务上可能有优势
   - 是现代Transformer架构的标准选择

4. **正则化的平衡**: 
   - 过度的dropout会损害性能
   - 需要根据具体架构调整正则化强度

### 高级实验与数据增强

#### 数据增强策略实现

本项目实现了多种现代数据增强技术：

**1. Cutout实现:**
```python
class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask
```

**2. AutoAugment策略:**
实现了CIFAR-10专门的AutoAugment策略，包括随机仿射变换、颜色抖动等。

**3. CutMix技术:**
混合不同图像的patch并相应调整标签，提高模型的泛化能力。

**数据增强效果分析:**
从可视化结果可以看出，不同的数据增强技术各有特点：
- **基础增强**: 随机裁剪和翻转，保持图像完整性
- **ColorJitter**: 改变颜色分布，增强颜色不变性
- **Rotation**: 旋转变换，增强几何不变性
- **Cutout**: 随机遮挡，模拟部分遮挡情况
- **AutoAugment**: 学习到的最优增强策略组合

### 优化器策略

**多种优化器对比：**

```python
optimizers_to_test = {
    'SGD': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    'Adam': lambda p: optim.Adam(p, lr=0.001),
    'AdamW': lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01),
    'RMSprop': lambda p: optim.RMSprop(p, lr=0.001),
    'CustomSGD': lambda p: CustomSGD(p, lr=0.01, momentum=0.9, nesterov=True)
}
```

**自定义SGD优化器：**

```python
class CustomSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0, nesterov=True):
        # 实现带Nesterov动量和L2正则化的SGD
```

- 从图中可以看出，Adam 和 AdamW 在早期阶段具有更快的收敛速度，并取得了较好的最终准确率；SGD 需要更长时间稳定收敛；RMSprop 效果最差，可能是超参数未调优；自定义 SGD 效果略优于原始 SGD。AdamW 综合表现最好。

![优化器对比图](./figs/optimizer_performance.png)


#### 损失函数实验

1. Focal Loss实验:

- 针对难分类样本加权
- 在类别不平衡时效果显著
- γ=2时取得最佳效果

2. 标签平滑实验:
- 防止模型过度自信
- 提升模型校准性能
- 平滑参数α=0.1时效果最佳

3. Center Loss实验:
- 增强同类特征聚类
- 结合交叉熵损失使用
---

## 三、Batch Normalization 分析与优化实验

### 1 BatchNorm 对训练效果的影响
通过对比 `VGG_A` 与 `VGG_A_BatchNorm` 的训练过程发现：
- 训练速度对比:
  - 标准VGG: 收敛慢，需要仔细调节学习率
  - 带BN的VGG: 收敛速度提升2倍，允许更高学习率（0.02 vs 0.01）

- 稳定性指标:
  - 标准VGG: 损失方差大，对初始化敏感
  - 带BN的VGG: 训练一致，对超参数不敏感

- 从训练曲线图可以观察到:
  - 损失曲线: BN版本下降更快且更平滑
  - 训练准确率: BN版本收敛到更高水平
  - 验证准确率: BN版本泛化性能更好


#### 1.2层级激活分析

层级激活统计分析：

**激活均值分析:**
- 标准VGG: 激活分布不均匀，某些层激活过大或过小
- 带BN的VGG: 激活分布更加均匀，有利于信息传播

**死神经元比例分析:**
从死神经元比例图可以看出：
- 标准VGG在深层网络中死神经元比例很高（>90%）
- BN显著降低了死神经元比例，保持网络活跃性
- 特别是在网络中后段，BN的效果更加明显

#### 1.3性能提升量化

| 指标 | 标准VGG | VGG + BN | 改善幅度 |
|------|---------|----------|----------|
| 最终验证准确率 | 86.43% | 89.54% | +3.11% |
| 训练稳定性 | 中等 | 高 | 定性改善 |
| 收敛速度 | 基准 | 2倍加快 | 定量改善 |
| 学习率容忍度 | 低 | 高 | 定性改善 |
| 死神经元比例 | 45.2% | 18.7% | 降低58.6% |

### 2 BN优化实现分析

#### 2.1 损失景观分析实现

通过实现了"梯度方向上的最大差异"分析来量化BN对优化景观的影响：

```python
def compute_max_gradient_difference(model, train_loader, criterion, 
                                   num_steps=10, max_distance=0.5):
    """
    测量梯度在不同距离上的最大差异
    这个函数实现了"Maximum difference in gradient over the distance"的要求
    """
    model.eval()
    distances = np.linspace(0, max_distance, num_steps)
    max_grad_diffs = []
    
    # 获取一个batch的数据
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    # 计算原始点的梯度
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # 保存原始梯度和参数
    original_grads = []
    original_params = []
    for param in model.parameters():
        if param.grad is not None:
            original_grads.append(param.grad.clone())
            original_params.append(param.data.clone())
    
    # 对每个距离计算梯度差异
    for dist in distances[1:]:  # 跳过0
        # 在梯度方向上移动
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    # 归一化梯度方向
                    grad_norm = original_grads[i].norm()
                    if grad_norm > 0:
                        param.data = original_params[i] + dist * original_grads[i] / grad_norm
        
        # 计算新位置的梯度
        model.zero_grad()
        outputs = model(inputs)
        loss_new = criterion(outputs, labels)
        loss_new.backward()
        
        # 计算最大梯度差异
        max_diff = 0
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                diff = torch.max(torch.abs(param.grad - original_grads[i])).item()
                max_diff = max(max_diff, diff)
        
        max_grad_diffs.append(max_diff)
        
        # 恢复原始参数
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if param.grad is not None:
                    param.data = original_params[i].clone()
    
    return distances[1:], max_grad_diffs
```

**实验结果分析:**

从梯度距离最大差异图可以清楚看出：

1. **标准VGG的梯度不稳定性:**
   - 蓝色曲线显示梯度差异随距离急剧增长
   - 在距离0.5处，最大梯度差异达到0.78
   - 说明局部线性近似很快失效

2. **BN的梯度稳定化效果:**
   - 红色曲线显示梯度差异增长缓慢
   - 在相同距离处，最大梯度差异仅为0.18
   - 局部线性近似在更大范围内有效

**分析结果:**

- 标准VGG平均最大梯度差异: 0.2841
- BatchNorm VGG平均最大梯度差异: 0.0892  
- 改善比例: 68.6%的梯度不稳定性降低
- 
![梯度差异图](./figs/maximum_gradient_difference.png)


#### 2.2梯度预测性分析

对梯度预测性测量，评估在梯度方向移动小步长后梯度的变化：

```python
def compute_gradient_predictiveness(model, train_loader, criterion, 
                                   steps=50, alpha=0.1):
    """计算梯度预测性 - 测量梯度的稳定性"""
    model.eval()
    gradient_errors = []
    
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= steps:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 计算原始点的梯度
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 保存原始梯度
        original_grads = []
        for param in model.parameters():
            if param.grad is not None:
                original_grads.append(param.grad.clone())
        
        # 在梯度方向上移动一小步
        with torch.no_grad():
            idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    param.data += alpha * original_grads[idx]
                    idx += 1
        
        # 计算新位置的梯度
        model.zero_grad()
        outputs = model(inputs)
        loss_new = criterion(outputs, labels)
        loss_new.backward()
        
        # 计算梯度差异
        grad_diff = 0
        idx = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_diff += torch.norm(param.grad - original_grads[idx]).item()
                idx += 1
        
        gradient_errors.append(grad_diff)
        
        # 恢复原始参数
        with torch.no_grad():
            idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= alpha * original_grads[idx]
                    idx += 1
    
    return gradient_errors
```


#### 2.3 梯度平滑度分析

Lipschitz平滑度分析：

```python
def compute_gradient_smoothness(model, train_loader, criterion, steps=50):
    """计算梯度平滑度 - 测量梯度的利普希茨连续性"""
    model.eval()
    smoothness_values = []
    
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= steps:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        
        # 生成随机扰动
        noise = torch.randn_like(inputs) * 0.01
        inputs_perturbed = inputs + noise
        
        # 计算原始梯度和扰动后梯度
        # ... [实现细节]
        
        # 计算梯度差异与输入差异的比值（Lipschitz常数的估计）
        grad_diff = torch.norm(grads_original - grads_perturbed)
        input_diff = torch.norm(noise.view(batch_size, -1), dim=1).mean()
        
        smoothness = (grad_diff / input_diff).item()
        smoothness_values.append(smoothness)
    
    return smoothness_values
```

**平滑度分析结果:**
- 标准VGG平均梯度平滑度: 2.8471
- BatchNorm VGG平均梯度平滑度: 1.2359

![梯度平滑度对比图](./figs/gradient_smoothness.png)


#### 2.4 深度网络分析

**层级激活统计分析:**

深入的层级分析：

```python
def analyze_layer_activations(model, data_loader, num_batches=5):
    """分析每层的激活值统计特性"""
    model.eval()
    activation_stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {
                    'mean': [],
                    'std': [],
                    'dead_neurons': []
                }
            
            # 计算激活统计
            act = output.detach()
            activation_stats[name]['mean'].append(act.mean().item())
            activation_stats[name]['std'].append(act.std().item())
            # 计算"死亡"神经元比例（激活值接近0）
            dead_ratio = (act.abs() < 0.01).float().mean().item()
            activation_stats[name]['dead_neurons'].append(dead_ratio)
        
        return hook
    
    # 注册hooks并收集数据
    # ... [实现细节]
    
    return activation_stats
```

**激活分析结果:**
从激活统计图可以看出：BN 模型在所有层中平均激活值波动更小，且“死亡神经元”（输出恒为0的神经元）比例更低，特别是在中后层结构中效果尤为显著。BN保持了更均匀的激活分布。

![各层死亡神经元比例](./figs/dead_neuron_ratio_by_layer.png)


**梯度健康度分析:**

```python
def analyze_gradient_health(model, data_loader, criterion):
    """分析梯度流的健康程度"""
    # 计算梯度统计
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            gradient_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'vanishing': (grad.abs() < 1e-7).float().mean().item(),
                'exploding': (grad.abs() > 1.0).float().mean().item()
            }
    
    return gradient_stats
```

**梯度健康度结果:**
- **梯度消失比例:**
  - 标准VGG: 15.2%
  - BatchNorm VGG: 3.8%
- **梯度爆炸比例:**
  - 标准VGG: 8.1%  
  - BatchNorm VGG: 1.2%

#### 2.2.4 损失地形可视化

**已实现损失地形分析：**

使用不同学习率训练模型，可视化损失变化：
- 学习率范围：[0.001, 0.005, 0.01, 0.05]
- 填充区域显示损失变化范围
- BatchNorm模型显示更平滑的损失景观

![landscape](./figs/loss_landscape.png)
![land_vgg](./figs/standard_vgg_loss_landscape.png)
![land_vggb](./figs/vgg_with_batchnorm_loss_landscape.png)


## 四、结论

- BatchNorm 不仅提升准确率，还能显著改善优化路径的平滑性
- 梯度相关可视化（预测性、平滑度、流动）验证了 BN 的优化优势
- 残差连接亦在较深模型中发挥了正向作用
- 通过 LR Finder 等工具可以提升训练效率与稳定性

---