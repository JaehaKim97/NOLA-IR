from typing import overload, Literal
import re

from PIL import Image
import torch

try:
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
    )
    from llava.utils import process_images_pad_torch_gpu, process_images_pad_pil

    LLAVA_AVAILABLE = True
except Exception as e:
    print(f"failed to import llava, error: {e}")
    LLAVA_AVAILABLE = False


try:
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram import get_transform

    RAM_AVAILABLE = True
except Exception as e:
    print(f"failed to import ram, error: {e}")
    RAM_AVAILABLE = False


class Captioner:

    def __init__(self, device: torch.device) -> "Captioner":
        self.device = device

    @overload
    def __call__(self, image: Image.Image) -> str: ...


class EmptyCaptioner(Captioner):

    def __call__(self, image: Image.Image) -> str:
        return ""


class LLaVACaptioner(Captioner):

    def __init__(
        self,
        device: torch.device,
        llava_bit: Literal["16", "8", "4"],
        qs: str = "Please give me a very short description of this image."
    ) -> "LLaVACaptioner":
        super().__init__(device)
        if llava_bit == "16":
            load_4bit, load_8bit = False, False
        elif llava_bit == "8":
            load_4bit, load_8bit = False, True
        else:
            load_4bit, load_8bit = True, False

        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        device_map = {"": device}
        self.tokenizer, self.model, self.image_processor, context_len = (
            load_pretrained_model(
                model_path,
                None,
                model_name,
                device=device,
                device_map=device_map,
                load_4bit=load_4bit,
                load_8bit=load_8bit,
            )
        )
        self.model.eval()

        # qs = "Please give me a very short description of this image."
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        self.prompt = conv.get_prompt()

        # self.temperature = 0
        # self.top_p = None
        # self.num_beams = 1
        # self.max_new_tokens = 512
        self.temperature = 0.3
        self.top_p = 0.9
        self.num_beams = 1
        self.max_new_tokens = 80

    @torch.no_grad()
    def __call__(self, image, convert_pil=False) -> str:
        if isinstance(image, Image.Image):
            batch_size = 1
            images = [image]
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images, self.image_processor, self.model.config
            ).to(self.device, dtype=torch.float16)
        else:
            batch_size, _, H0, W0 = image.shape
            image_sizes = [(W0, H0)] * batch_size
            if convert_pil:
                images_tensor = process_images_pad_pil(
                    image, self.image_processor, process_images, self.model.config
                ).to(self.device, torch.float16)
            else:
                images_tensor = process_images_pad_torch_gpu(
                    image, self.image_processor, device=self.device, out_dtype=torch.float16
                )
        input_ids = (
            tokenizer_image_token(
                self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )
        output_ids = self.model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        res = [s.strip() for s in outputs]
        return res


class RAMCaptioner(Captioner):

    def __init__(self, device: torch.device) -> Captioner:
        super().__init__(device)
        image_size = 384
        transform = get_transform(image_size=image_size)
        pretrained = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
        model = ram_plus(pretrained=pretrained, image_size=image_size, vit="swin_l")
        model.eval()
        model = model.to(device)

        self.transform = transform
        self.model = model

    def __call__(self, image: Image.Image) -> str:
        image = self.transform(image).unsqueeze(0).to(self.device)
        res = inference(image, self.model)
        # res[0]: armchair | blanket | lamp | ...
        # res[1]: 扶手椅  | 毯子/覆盖层 | 灯  | ...
        return res[0].replace(" | ", ", ")
