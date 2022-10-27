CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe.py --checkpoint runs/crepe/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe-mdb.py --checkpoint runs/crepe-mdb/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe-ptdb.py --checkpoint runs/crepe-ptdb/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe-1440.py --checkpoint runs/crepe-1440/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe-original-mdb.py --checkpoint runs/crepe-original-mdb/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/crepe-noblur.py --checkpoint runs/crepe-noblur/00250000.pt --gpu 0 &&
CUDA_VISIBLE_DEVICES=1 python -m penne.evaluate --config config/harmof0.py --checkpoint runs/harmof0/00250000.pt --gpu 0
