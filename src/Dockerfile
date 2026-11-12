FROM crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0@sha256:5e333ae965f02b4c80e4a5c89a9d33cae9ba54ffaea2f73217ffcd4b4a7fabc6

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    TORCH_HOME=/workspace/.cache/torch

RUN python -c "import sys,torch,torchvision; assert sys.version_info[:2] == (3,12); assert torch.__version__.split('+')[0] == '2.10.0'; assert torchvision.__version__.split('+')[0] == '0.25.0'; assert torch.version.cuda == '12.8'"

RUN python -m venv --without-pip --system-site-packages /opt/cxr-venv
ENV PATH="/opt/cxr-venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir uv==0.11.21
COPY src/requirements-tools.txt /opt/cxr/requirements-tools.txt
RUN uv pip install --python /opt/cxr-venv/bin/python --no-cache -r /opt/cxr/requirements-tools.txt

WORKDIR /workspace
