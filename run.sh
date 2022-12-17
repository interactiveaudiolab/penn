# Runs all experiments in the paper
# "Cross-domain Neural Pitch and Periodicity Estimation"

# Args
# $1 - list of indices of GPUs to use

# # Download datasets
# python -m penn.data.download

# # Setup and run 16 kHz experiments
# python -m penn.data.preprocess --config config/crepe.py
# python -m penn.partition
# python -m penn.train --config config/crepe.py --gpus $1
# python -m penn.train --config config/deepf0.py --gpus $1

# # Evaluate baselines at 16 kHz
# python -m penn.evaluate --gpu 0 --config config/dio.py
# python -m penn.evaluate --gpu 0 --config config/pyin.py
# python -m penn.evaluate \
#     --gpu 0 \
#     --method torchcrepe \
#     --config config/torchcrepe.py

# # Setup 8 kHz data
# python -m penn.data.preprocess

# # Run 8 kHz experiments
# python -m penn.train --config config/crepe++.py --gpus $1
# python -m penn.train --config config/deepf0++.py --gpus $1
# python -m penn.train --config config/fcnf0.py --gpus $1
# python -m penn.train --config config/fcnf0++.py --gpus $1

# # Train on individual datasets
# python -m penn.train \
#     --config config/fcnf0++-mdb.py \
#     --datasets mdb \
#     --gpus $1
# python -m penn.train \
#     --config config/fcnf0++-ptdb.py \
#     --datasets ptdb \
#     --gpus $1

# # Train ablations
# python -m penn.train --config config/fcnf0++-ablate-batchsize.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-earlystop.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-inputnorm.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-layernorm.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-loss.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-quantization.py --gpus $1
# python -m penn.train --config config/fcnf0++-ablate-unvoiced.py --gpus $1

# # Evaluate locally normal decoding
# python -m penn.evaluate \
#     --config config/fcnf0++-ablate-decoder.py \
#     --checkpoint runs/fcnf0++/00250000.pt \
#     --gpu $1

# Plot data and inference distributions
python -m penn.plot.density \
    --datasets mdb \
    --output_file results/mdb_on_mdb.jpg \
    --checkpoint runs/fcnf0++-mdb/00250000.pt \
    --gpu $1
python -m penn.plot.density \
    --datasets ptdb \
    --output_file results/mdb_on_ptdb.jpg \
    --checkpoint runs/fcnf0++-mdb/00250000.pt \
    --gpu $1
python -m penn.plot.density \
    --datasets ptdb \
    --output_file results/ptdb_on_ptdb.jpg \
    --checkpoint runs/fcnf0++-ptdb/00250000.pt \
    --gpu $1
python -m penn.plot.density \
    --datasets mdb \
    --output_file results/ptdb_on_mdb.jpg \
    --checkpoint runs/fcnf0++-ptdb/00250000.pt \
    --gpu $1
python -m penn.plot.density \
    --datasets mdb \
    --output_file results/both_on_mdb.jpg \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1
python -m penn.plot.density \
    --datasets ptdb \
    --output_file results/both_on_ptdb.jpg \
    --checkpoint runs/fcnf0++/00250000.pt \
    --gpu $1

# Plot voiced/unvoiced threshold landscape
# python -m penn.plot.threshold \
#     --names FCNF0++ "FCNF0++ (voiced only)" \
#     --evaluations fcnf0++-ablate-decoder fcnf0++-ablate-unvoiced \
#     --output_file results/threshold.jpg

# Plot pitch posteriorgram figures
# python -m penn.plot.logits \
#     --config config/fcnf0++.py \
#     --audio_file test/assets/gershwin.wav \
#     --output_file results/fcnf0++-gershwin.jpg \
#     --checkpoint runs/fcnf0++/00250000.pt \
#     --gpu $1
# Note - You will need to replace this checkpoint with the final checkpoint
#        that was produced during fcnf0 training
# python -m penn.plot.logits \
#     --config config/fcnf0.py \
#     --audio_file test/assets/gershwin.wav \
#     --output_file results/fcnf0-gershwin.jpg \
#     --checkpoint runs/fcnf0/00068000.pt \
#     --gpu $1
