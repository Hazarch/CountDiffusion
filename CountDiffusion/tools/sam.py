import numpy as np
import torch
import os

import torchvision
from PIL import Image, ImageDraw

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np

from omegaconf import OmegaConf

class sam:
    def __init__(self, config, device='cuda'):
        self.device = device
        self.model = self.load_model(config.comfig_file, config.grounded_checkpoint, self.device)
        self.predictor = SamPredictor(build_sam(checkpoint=config.sam_checkpoint).to(self.device))
        self.box_threshold = config.box_threshold
        self.text_threshold = config.text_threshold
        self.iou_threshold = config.iou_threshold

    def load_image(self, image_pil):
        # load image
        image_pil = image_pil.convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    def get_grounding_output(self, image, caption):
        
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = self.model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())
        
        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    @torch.no_grad()
    def __call__(self, image_pil:Image.Image, tags:list, save_file:str=None):
        image_pil, image = self.load_image(image_pil)
        all_tags = ""
        for tag in tags:
            all_tags = all_tags + tag + ", "
        all_tags = all_tags[:-2]
 
        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(image, all_tags)
      
        image = np.array(image_pil)
        self.predictor.set_image(np.array(image))

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        if save_file is not None:
            draw = ImageDraw.Draw(image_pil)
            for box in np.array(transformed_boxes.cpu()):
                # print(box)
                x1, y1, x2, y2 = box
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            image_pil.save(os.path.join(save_file, "bbox.png"))
            
        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        ret = {}
        for tag in tags:
            ret[tag] = []
     
        for label, mask in zip(pred_phrases, masks):
            item, _ = label.split('(')
            if item not in tags:
                for one_tag in tags:
                    if item in one_tag:
                        tag = one_tag
                        break
            else:
                tag = item
            ret[tag].append(mask)
        for key, value in ret.items():
            if value != []:
                ret[key] = torch.stack(value)
            print(f"\ndetect {key}: {len(value)}\n")

            if save_file is not None:
                for j, mask in enumerate(value):
                    mask_np = (mask.squeeze().cpu().numpy() > 0)
                    mask_scaled = (mask_np * 255).astype(np.uint8)
                    mask_color = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_WINTER)
                    save_path = os.path.join(save_file, f'{key}_{j}.png')
                    cv2.imwrite(save_path, mask_color)

                mask = (np.array(sum([mask.squeeze().cpu().numpy() for mask in value])) > 0)
                mask_scaled = (mask * 255).astype(np.uint8)
                mask_color = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_WINTER)
                save_path = os.path.join(save_file, f'{key}.png')
                cv2.imwrite(save_path, mask_color)

        return ret

if __name__ == "__main__":
    config = "configs/sam_config.yaml"
    config = OmegaConf.load(config).sam

    my_sam = sam(config)
    img_path = "/home/wpc/project/PixArt-sigma-cnt/tools/In a peaceful village, a mischievous child discovers a hidden cave containing a collection of two balloons, each inflated balloon showcasing a different historical era.jpg"
    tags = ["balloons"]

    img_pil = Image.open(img_path)
    out = my_sam(image_pil=img_pil, tags=tags, save_file="/home/wpc/project/PixArt-sigma-cnt/output/mid")
