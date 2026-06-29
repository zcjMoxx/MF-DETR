import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", category=UserWarning, mecssage="setting 'requires_grad=True' for frozen layer.*")

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = 'dataset/data_VisDrone.yaml'
    check_path(yaml_path)
    model = RTDETR('ultralytics/cfg/models/rt-detr/mf-detr-r18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=str(yaml_path),
                cache=True,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=8,
                device='0',
                # freeze=[], # freeze layers
                # resume='', # last.pt path
                project='runs/train',
                name='mfdetr-visdrone',
                patience=40,
                verbose=False
                )