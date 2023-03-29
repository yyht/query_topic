

conda create  -c conda-forge -y -p /data/albert.xht/ENV.docker.onnx python=3.8;
conda init bash
conda activate /data/albert.xht/ENV.docker.onnx

conda install --use-local /data/albert.xht/cudnn-8.0.5.39-hc0a50b0_1.tar.bz2

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch -c conda-forge 