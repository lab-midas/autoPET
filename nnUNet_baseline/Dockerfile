FROM python:3.9-slim



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

RUN mkdir -p /opt/algorithm/checkpoints/nnUNet/

# Store your weights in the container
#COPY --chown=algorithm:algorithm ../3d_fullres /opt/algorithm/checkpoints/nnUNet/
COPY --chown=algorithm:algorithm weights.zip /opt/algorithm/checkpoints/nnUNet/
RUN python -c "import zipfile; zipfile.ZipFile('/opt/algorithm/checkpoints/nnUNet/weights.zip').extractall('/opt/algorithm/checkpoints/nnUNet/')"

# nnUNet specific setup
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result


#RUN ln -sf /input/images /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs

ENV nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base"
ENV RESULTS_FOLDER="/opt/algorithm/checkpoints"


ENTRYPOINT python -m process $0 $@

