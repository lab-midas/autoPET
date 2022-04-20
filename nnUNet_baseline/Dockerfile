#FROM python:3.9-slim
FROM pytorch/pytorch


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict.py /opt/algorithm/

RUN mkdir -p /opt/algorithm/checkpoints/nnUNet/

# Store your weights in the container
COPY --chown=algorithm:algorithm weights.zip /opt/algorithm/checkpoints/nnUNet/
RUN python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/checkpoints/nnUNet/weights.zip').extractall('/opt/algorithm/checkpoints/nnUNet/')"

# nnUNet specific setup
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result

ENV nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base"
ENV RESULTS_FOLDER="/opt/algorithm/checkpoints"
ENV MKL_SERVICE_FORCE_INTEL=1


ENTRYPOINT python -m process $0 $@

