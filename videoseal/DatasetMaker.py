# %%
# %load_ext autoreload
# %autoreload 2
# %cd /home/teaching/Desktop/Grp_22/BiometricSEAL/videoseal/

# %%
import os
import omegaconf
from tqdm import tqdm
import gc
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import random
from videoseal.utils.display import save_img
from videoseal.utils import Timer
from videoseal.evals.full import setup_model_from_checkpoint
from videoseal.evals.metrics import bit_accuracy, psnr, ssim
from videoseal.augmentation import Identity, JPEG
from videoseal.modules.jnd import JND
to_tensor = torchvision.transforms.ToTensor()
to_pil = torchvision.transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# num_imgs = -1
assets_dir = "../train/"
output_dir = "apnaDataset"
base_output_dir = "outputs"
os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# %%
#write code for loadigng tensor from tensor.pt
tensor = torch.load("tensors.pt")


# %%
print(tensor[0].shape)

# %%
ckpt_path = "output/checkpoint025.pth"

# %%
model = setup_model_from_checkpoint(ckpt_path)
model.eval()
model.compile()
model.to(device)
model.blender.scaling_w *= 1

# %%


# %%
files = [f for f in os.listdir(assets_dir) if f.endswith(".png") or f.endswith(".jpg")]
files = [os.path.join(assets_dir, f) for f in files]
# files = files[:]
for file in tqdm(files, desc=f"Processing Images"):
        # load image
        imgs = Image.open(file, "r").convert("RGB")  # keep only rgb channels
        imgs = to_tensor(imgs).unsqueeze(0).float()

        # Watermark embedding
        # timer.start()
        custom_msg = random.choice(tensor)
        #get index of the tensor
        ind = 0
        for i in range(len(tensor)):
            if torch.equal(custom_msg, tensor[i]):
                ind = i
                break
            
        outputs = model.embed(imgs, msgs = custom_msg, is_video=False, lowres_attenuation=True)
        torch.cuda.synchronize()
        # print(f"embedding watermark  - took {timer.stop():.2f}s")

        # compute diff
        imgs_w = outputs["imgs_w"]  # b c h w
        msgs = outputs["msgs"]  # b k
        diff = imgs_w - imgs
        diff = 25 * diff.abs()

        # save
        # timer.start()
        base_save_name = os.path.join(output_dir, os.path.basename(file).replace(".png", ""))
        # print(base_save_name)
        # save_img(imgs[0], f"{base_save_name}_ori.png")
        save_img(imgs_w[0], f"{base_save_name}_wm.png")
        # save_img(diff[0], f"{base_save_name}_diff.png")

        with open("Dataset.txt", 'a') as f:
                f.write(f"{base_save_name}_wm.png;{ind};\n")

        # Metrics
        imgs_aug = imgs_w
        outputs = model.detect(imgs, is_video=False)
        metrics = {
            "file": file,
            "bit_accuracy": bit_accuracy(
                outputs["preds"][:, 1:],
                msgs
            ).nanmean().item(),
            "psnr": psnr(imgs_w, imgs).item(),
            "ssim": ssim(imgs_w, imgs).item()
        }

# %%


# %%



