# Runs all experiments in the paper
# "Cross-domain Neural Pitch and Periodicity Estimation"

# Args
# $1 - list of indices of GPUs to use

# Download datasets
# python -m penne.data.download

# Setup and run 16 kHz experiments
# python -m penne.data.preprocess --config config/crepe.py
# python -m penne.partition
# python -m penne.train --config config/crepe.py --gpus $1
# python -m penne.train --config config/deepf0.py --gpus $1

# Evaluate baselines at 16 kHz
python -m penne.evaluate --gpu 0 --config config/dio.py
python -m penne.evaluate --gpu 0 --config config/pyin.py
python -m penne.evaluate --gpu 0 --config config/pyin-viterbi.py
python -m penne.evaluate \
    --gpu 0 \
    --method torchcrepe \
    --config config/torchcrepe.py

# Setup 8 kHz data
python -m penne.data.preprocess

# Run 8 kHz experiments
python -m penne.train --config config/crepe++.py --gpus $1
python -m penne.train --config config/deepf0++.py --gpus $1
python -m penne.train --config config/fcnf0.py --gpus $1
python -m penne.train --config config/fcnf0++.py --gpus $1

# Train on individual datasets
python -m penne.train \
    --config config/fcnf0++-mdb.py \
    --datasets mdb \
    --gpus $1
python -m penne.train \
    --config config/fcnf0++-ptdb.py \
    --datasets ptdb \
    --gpus $1

# Train ablations
python -m penne.train --config config/fcnf0++-ablate-batchsize.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-earlystop.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-inputnorm.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-layernorm.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-loss.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-quantization.py --gpus $1
python -m penne.train --config config/fcnf0++-ablate-unvoiced.py --gpus $1

# Evaluate decoding methods
python -m penne.evaluate \
    --config config/fcnf0++-dither.py \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1
python -m penne.evaluate \
    --config config/fcnf0++-weighted.py \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1

# Dataset and true positive density figures
python -m penne.plot.density \
    --true_datasets mdb \
    --inference_datasets ptdb \
    --output_file results/mdb_on_ptdb.jpg \
    --checkpoint runs/fcnf0++-ptdb/00250000.pt \
    --gpu $1
python -m penne.plot.density \
    --true_datasets ptdb \
    --inference_datasets mdb \
    --output_file results/ptdb_on_mdb.jpg \
    --checkpoint runs/fcnf0++-mdb/00250000.pt \
    --gpu $1
python -m penne.plot.density \
    --true_datasets mdb ptdb \
    --inference_datasets mdb \
    --output_file results/both_on_mdb.jpg \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1
python -m penne.plot.density \
    --true_datasets mdb ptdb \
    --inference_datasets mdb \
    --output_file results/both_on_ptdb.jpg \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1

# Voiced/unvoiced threshold landscape
python -m penne.plot.threshold \
    --output_file results/threshold.jpg \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1

# Perform inference on test audio
python -m penne \
    --config config/crepe.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/crepe-gershwin.pt \
    --checkpoint runs/crepe/00126500.pt \
    --gpu $1
python -m penne \
    --config config/crepe++.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/crepe++-gershwin.pt \
    --checkpoint runs/crepe++/00250000.pt \
    --gpu $1
python -m penne \
    --config config/deepf0.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/deepf0-gershwin.pt \
    --checkpoint runs/deepf0/00099000.pt \
    --gpu $1
python -m penne \
    --config config/deepf0++.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/deepf0++-gershwin.pt \
    --checkpoint runs/deepf0++/00250000.pt \
    --gpu $1
python -m penne \
    --config config/fcnf0.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/fcnf0-gershwin.pt \
    --checkpoint runs/fcnf0/00142000.pt \
    --gpu $1
python -m penne \
    --config config/fcnf0++.py \
    --audio_files test/assets/gershwin.wav \
    --output_prefixes results/fcnf0++-gershwin.pt \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1

# Pitch posteriorgram figures
python -m penne.plot.logits \
    --config config/crepe.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/crepe-gershwin.jpg \
    --pitch_file results/crepe-gershwin.pt \
    --checkpoint runs/crepe/00126500.pt \
    --gpu $1
python -m penne.plot.logits \
    --config config/crepe++.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/crepe++-gershwin.jpg \
    --pitch_file results/crepe++-gershwin.pt \
    --checkpoint runs/crepe++/00250000.pt \
    --gpu $1
python -m penne.plot.logits \
    --config config/deepf0.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/deepf0-gershwin.jpg \
    --pitch_file results/deepf0-gershwin.pt \
    --checkpoint runs/deepf0/00099000.pt \
    --gpu $1
python -m penne.plot.logits \
    --config config/deepf0++.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/deepf0++-gershwin.jpg \
    --pitch_file results/deepf0++-gershwin.pt \
    --checkpoint runs/deepf0++/00250000.pt \
    --gpu $1
python -m penne.plot.logits \
    --config config/fcnf0.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/fcnf0-gershwin.jpg \
    --pitch_file results/fcnf0-gershwin.pt \
    --checkpoint runs/fcnf0/00142000.pt \
    --gpu $1
python -m penne.plot.logits \
    --config config/fcnf0++.py \
    --audio_file test/assets/gershwin.wav \
    --output_file results/fcnf0++-gershwin.jpg \
    --pitch_file results/fcnf0++-gershwin.pt \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1
