python -m penne.data.download && \
python -m penne.data.preprocess && \
python -m penne.partition && \
python -m penne.train --config config/crepe.py --gpus 0 1 &&
python -m penne.train --config config/crepe++.py --gpus 0 1 &&
python -m penne.train --config config/deepf0.py --gpus 0 1 &&
python -m penne.train --config config/deepf0++.py --gpus 0 1 &&
python -m penne.train --config config/fcnc0.py --gpus 0 1 &&
python -m penne.train --config config/fcnf0++.py --gpus 0 1 &&
python -m penne.train --config config/harmof0.py --gpus 0 1 &&
python -m penne.train --config config/harmof0++.py --gpus 0 1
# TODO
