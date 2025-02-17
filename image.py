import torch
import comfy
import numpy as np
import math
from PIL import Image
from torchvision.transforms.functional import to_pil_image

class FastAlphaCropper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 500}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, image, padding: int = 0):
        cropped_images = []
        cropped_masks = []

        # 遍历每个批次的图像
        for img in image:
            # 提取Alpha通道（假设RGBA格式）
            alpha = img[..., 3]

            height = img.shape[0]
            width = img.shape[1]
            # 创建有效区域掩码
            mask = (alpha > 0.01)

            # 寻找有效区域边界
            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)

            ymin, ymax = self._find_boundary(rows)
            xmin, xmax = self._find_boundary(cols)

            # 处理全透明情况
            if ymin is None or xmin is None:
                cropped_images.append(img)
                cropped_masks.append(torch.zeros_like(alpha))
                continue

            # 添加padding并限制边界
            ymin = max(0, ymin - padding)
            ymax = min(height, ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(width, xmax + padding)

            # 执行裁剪
            cropped = img[ymin:ymax, xmin:xmax, :4]  # 保留RGB通道
            cropped_mask = alpha[ymin:ymax, xmin:xmax]

            cropped_images.append(cropped)
            cropped_masks.append(cropped_mask)

        return cropped_images, cropped_masks
    def _find_boundary(self, arr):
        nz = torch.nonzero(arr)
        if nz.numel() == 0:
            return (None, None)
        return (nz[0].item(), nz[-1].item() + 1)


class ShrinkImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["scale", "pixels"], {"default": "scale"}),
                "resize_algorithm": (list(resize_algorithms.keys()), {"default": "LANCZOS"})
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1}),
                "height": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shrink_image"
    CATEGORY = "image/processing"

    def calculate_scale(self, img, mode, scale=None, width=None, height=None):
        if mode == "scale":
            return scale
        else:
            img_width, img_height = img.size
            width = min(width, img_width)
            height = min(height, img_height)
            scale_x = width / img_width
            scale_y = height / img_height
            return min(scale_x, scale_y)

    def shrink_image_with_scale(self, img, scale, algorithm):
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return img.resize((new_width, new_height), algorithm)

    def shrink_image(self, image, mode, resize_algorithm, scale=None, width=None, height=None):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        algorithm = resize_algorithms[resize_algorithm]

        output_images = []
        for img in image:
            print('img.shape=', img.shape)
            img = to_pil_image(img.permute(2, 0, 1))
            scale = self.calculate_scale(img, mode, scale, width, height)
            resized_img = self.shrink_image_with_scale(img, scale, algorithm)
            resized_img_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_img_np = torch.from_numpy(resized_img_np)
            output_images.append(resized_img_np)

        return (output_images,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "FastAlphaCropper": FastAlphaCropper,
    "ShrinkImage": ShrinkImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastAlphaCropper": "Fast Alpha Cropper",
    "ShrinkImage": "Shrink Image"
}