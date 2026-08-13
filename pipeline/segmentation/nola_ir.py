import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.edtr import EDTRPipeline


class NOLAIRPipeline(EDTRPipeline):

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_unet()
        self.init_lora()
        self.init_seg_model()

    def init_optimizers(self):
        if not self.args.eval_only:
            opt_config = self.cfg.train.optimizer.nolair
            if opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for NOLA-IR")

            self.gen_params = list(self.lora_params)
            self.opt_gen = optimizer(self.gen_params, **opt_config.kwargs)
            self.sch_gen = CosineAnnealingLR(self.opt_gen, T_max=self.cfg.train.train_steps, eta_min=1e-7)

            opt_config = self.cfg.train.optimizer.seg_model
            if opt_config.type.lower() == "sgd":
                optimizer = torch.optim.SGD
            elif opt_config.type.lower() == "adamw":
                optimizer = torch.optim.AdamW
            else:
                raise NotImplementedError(f"{opt_config.type} is Not supported optimizer for seg_model")

            self.seg_model_params = list(filter(lambda p: p.requires_grad, self.seg_model.parameters()))
            self.opt_seg_model = optimizer(self.seg_model_params, **opt_config.kwargs)
            self.sch_seg_model = CosineAnnealingLR(self.opt_seg_model, T_max=self.cfg.train.train_steps, eta_min=1e-7)

    def prepare_all(self):
        attrs = ["res_model", "vae", "text_encoder", "unet", "seg_model", "seg_hq_model", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "opt_seg_model", "sch_seg_model", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def partial_diff_and_sample(self, x, c_img, c_txt, t=None, mode="eval"):
        for i, step in enumerate(sorted(self.timesteps, reverse=True)):
            if mode == "eval":
                t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)

            eps = self.unet(x, t, encoder_hidden_states=c_txt).sample

            index = self.t2index(t)
            x0_hat = self.spaced_sampler._predict_xstart_from_eps(x, index, eps)
            x = x0_hat
            if mode == "train":
                break
        return x

