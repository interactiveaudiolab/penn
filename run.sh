# Runs the whole paper
# Training assumes two GPUs in slots 0 and 1

# Download datasets
python -m penne.data.download && \

# Setup and run 16 kHz experiments
python -m penne.data.preprocess --config config/crepe.py && \
python -m penne.partition && \
python -m penne.train --config config/crepe.py --gpus 0 1 && \
python -m penne.train --config config/deepf0.py --gpus 0 1 && \
# python -m penne.train --config config/harmof0.py --gpus 0 1 && \

# Evaluate baselines at 16 kHz
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --method pyin --config config/pyin.py && \
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate \
    --gpu 0 \
    --method pyin \
    --config config/pyin-viterbi.py && \
python -m penne.evaluate \
    --gpu 0 \
    --method torchcrepe \
    --config config/torchcrepe.py && \

# Train on individual datasets
python -m penne.train \
    --config config/fcnf0++-mdb.py \
    --datasets mdb \
    --gpus 0 1 && \
python -m penne.train \
    --config config/fcnf0++-ptdb.py \
    --datasets ptdb \
    --gpus 0 1 && \

# Setup 8 kHz data
python -m penne.data.preprocess && \

# Run 8 kHz experiments
python -m penne.train --config config/crepe++.py --gpus 0 1 && \
python -m penne.train --config config/deepf0++.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++.py --gpus 0 1 && \
# python -m penne.train --config config/harmof0++.py --gpus 0 1 && \

# Train ablations
python -m penne.train --config config/fcnf0++-ablate-batchsize.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-earlystop.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-inputnorm.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-layernorm.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-loss.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-quantization.py --gpus 0 1 && \
python -m penne.train --config config/fcnf0++-ablate-unvoiced.py --gpus 0 1 && \

# Evaluate decoding methods
python -m penne.evaluate --config config/fcnf0++-dither.py --gpu 0 && \
python -m penne.evaluate --config config/fcnf0++-weighted.py --gpu 0 && \

# Aggregate evaluation into tables
python -m penne.evaluate.analyze

# TODO - add documentation of figure parameters
