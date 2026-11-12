# 已验证的官方容器环境

2026-10-26，本机Docker中实际运行并通过CUDA矩阵运算检查：

| 项目 | 实测 |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| CUDA运行时 | 12.8 |
| cuDNN | 9.10.2（91002） |
| GPU | NVIDIA GeForce RTX 5060 Ti，8GB |
| 主机驱动 | 610.88，未修改 |

官方基础镜像：

```text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0
sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6
```

`src/Dockerfile`固定上述digest，派生镜像名`cxr-phase1:official`。原镜像系统Python受PEP668保护，且没有ensurepip；派生镜像在`/opt/cxr-venv`创建继承官方包的独立虚拟环境，用uv补充数据处理和验收工具，不替换torch/torchvision。

完整运行时包清单见`container-packages.txt`，GPU检查证据在`.work/phase1/container-environment.json`。本机已有Anaconda及其Conda环境不改动，Windows原有环境不作为本项目正式运行环境。

运行入口`src/run_docker.ps1`将工作区映射到`/workspace`，原始数据目录另作只读挂载。GPU型号仍与主办方T4不同；本地延迟不能替代T4正式验收。

验证命令：

```powershell
powershell -NoProfile -File src/run_docker.ps1 check
powershell -NoProfile -File src/run_docker.ps1 freeze
powershell -NoProfile -File src/run_docker.ps1 test
powershell -NoProfile -File src/run_docker.ps1 lint
powershell -NoProfile -File src/run_docker.ps1 typecheck
```

CUDA运行时和框架版本对齐即可保留训练/推理兼容性基准；主机驱动不必降级到官方最低版本。参见[PyTorch版本安装说明](https://pytorch.org/get-started/previous-versions/)、[NVIDIA CUDA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[Docker Windows GPU支持](https://docs.docker.com/desktop/features/gpu/)。
