import os
from pathlib import Path

import cv2
import torch
import torchvision
import torch.optim

import numpy as np
from PIL import Image

# import net
from preprocessing.dehaze import net


def demo(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

    clean_image = dehaze_net(data_hazy)
    torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])


def dehaze_image(image_path):
    # --- 1. 读图并转为 [0,1] numpy --- #
    data_hazy_pil = Image.open(image_path).convert("RGB")
    data_hazy_np = np.asarray(data_hazy_pil) / 255.0  # (H,W,3) RGB

    # --- 2. numpy → torch tensor → GPU --- #
    data_hazy = torch.from_numpy(data_hazy_np).float()  # (H,W,3)
    data_hazy = data_hazy.permute(2, 0, 1).unsqueeze(0).cuda()  # (1,3,H,W)

    # --- 3. load model --- #
    dehaze_net = net.dehaze_net().cuda()
    ckpt_path = Path(__file__).parent.resolve() / 'snapshots/dehazer.pth'
    dehaze_net.load_state_dict(torch.load(ckpt_path))
    dehaze_net.eval()

    # --- 4. forward --- #
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)

    # --- 5. 转回 numpy --- #
    # data_hazy: (1,3,H,W) GPU → CPU → numpy
    data_hazy_np = data_hazy[0].permute(1, 2, 0).cpu().numpy()
    clean_image_np = clean_image[0].permute(1, 2, 0).cpu().numpy()

    # --- 6. clip ------------------------------------------------------------------- #
    data_hazy_np = np.clip(data_hazy_np, 0, 1)
    clean_image_np = np.clip(clean_image_np, 0, 1)

    # --- 7. 转为 OpenCV 用的 uint8 --- #
    data_hazy_cv = (data_hazy_np * 255).astype(np.uint8)
    clean_image_cv = (clean_image_np * 255).astype(np.uint8)

    # OpenCV 的通道是 BGR
    data_hazy_cv = cv2.cvtColor(data_hazy_cv, cv2.COLOR_RGB2BGR)
    clean_image_cv = cv2.cvtColor(clean_image_cv, cv2.COLOR_RGB2BGR)

    return data_hazy_cv, clean_image_cv


if __name__ == '__main__':
    test_dir = 'test_data'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    image_dir = Path(test_dir)
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [(str(p)) for p in image_dir.iterdir() if p.suffix.lower() in exts]
    sorted_files = sorted(image_files)

    for i, path in enumerate(sorted_files, start=1):
        try:
            data_hazy, clean_image = dehaze_image(path)
            save_path = os.path.join(output_dir, f'dehazed_{Path(path).name}')
            cv2.imwrite(save_path, clean_image)

            save_path = os.path.join(output_dir, f'combined_{Path(path).name}')
            combined = np.hstack((data_hazy, clean_image))
            cv2.imwrite(save_path, combined)
        except Exception as e:
            print(e)
        print(i, path, "done!")
