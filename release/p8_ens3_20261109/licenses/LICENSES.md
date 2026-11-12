# 第三方许可证全文与来源

本交付包推理阶段只依赖四个第三方库，均为宽松许可证、非 NC、非 gated。

## 1. PyTorch — BSD-3-Clause

来源：https://github.com/pytorch/pytorch/blob/main/LICENSE
版权：Copyright (c) 2016-     Facebook, Inc (Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan)

```
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 2. torchvision — BSD-3-Clause

来源：https://github.com/pytorch/vision/blob/main/LICENSE
版权：Copyright (c) Soumith Chintala 2016
正文与第 1 节相同（BSD-3-Clause）。

## 3. NumPy — BSD-3-Clause

来源：https://github.com/numpy/numpy/blob/main/LICENSE.txt
版权：Copyright (c) 2005-2024, NumPy Developers
正文与第 1 节相同（BSD-3-Clause）。

## 4. Pillow — HPND

来源：https://github.com/python-pillow/Pillow/blob/main/LICENSE
版权：Copyright (c) 2010-2024 by the PIL contributors
HPND（Historical Permission Notice and Disclaimer），允许商用、修改与再分发，需保留版权声明。

## 5. 预训练权重（EfficientNet-B0，ImageNet-1k）

- 来源：torchvision 官方 `EfficientNet_B0_Weights.IMAGENET1K_V1`
- URL：https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
- 训练数据：ImageNet-1k（自然图像，非医学）
- 许可证：随 torchvision 权重发布，允许赛事使用与再分发；上游数据条款见 https://image-net.org/
- 本包内 `pretrain_model/README.md` 有完整来源、哈希与使用声明。

## 6. 本队代码

`code/predict.py`、`code/cost.py` 由本队编写，随包提供用于赛事复核。
