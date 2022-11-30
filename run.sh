# Runs the whole paper

# Args
# $1 - list of indices of GPUs to use

# Download datasets
# python -m penne.data.download

# Setup and run 16 kHz experiments
# python -m penne.data.preprocess --config config/crepe.py
# python -m penne.partition
python -m penne.train --config config/crepe.py --gpus $1
python -m penne.train --config config/deepf0.py --gpus $1
# python -m penne.train --config config/harmof0.py --gpus $1

# Evaluate baselines at 16 kHz
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --config config/dio-16k.py
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --config config/pyin-16k.py
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --config config/pyin-viterbi-16k.py
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
# python -m penne.train --config config/harmof0++.py --gpus $1

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
python -m penne.evaluate --config config/fcnf0++-dither.py --gpu 0
python -m penne.evaluate --config config/fcnf0++-weighted.py --gpu 0

# Evaluate baselines at 8 kHz
python -m penne.evaluate --gpu 0 --config config/dio.py
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --config config/pyin.py
# MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m penne.evaluate --gpu 0 --config config/pyin-viterbi.py

# Aggregate evaluation into tables
python -m penne.evaluate.analyze

# TODO - add documentation of figure parameters
