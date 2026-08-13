import copy

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.segmentation.nola_ir import NOLAIRPipeline
from utils.common import wavelet_reconstruction
from utils.segmentation import mask2rgb


class NOLAIRTPGANPipeline(NOLAIRPipeline):

    def init_models(self):
        self.init_timesteps()
        self.init_scheduler()
        self.init_res_model(train=False)
        self.init_text_models()
        self.init_vae()
        self.init_unet()
        self.init_lora()
        self.init_unet_ref()
        self.init_seg_model()
        self.init_disc()

    def init_unet_ref(self):
        self.unet_ref = copy.deepcopy(self.unet).requires_grad_(False)
        if not self.args.eval_only:
            load_path = self.cfg.train.resume.get("lora_unet_ref", False)
            if load_path:
                load_path, strict = self.get_model_load_path("lora_unet_ref")
                m, u = self.unet_ref.load_state_dict(torch.load(load_path), strict=False)
                self.log(f"LoRA-UNet REF: Load from: {load_path}, unexpected keys: {u}")
        self.unet_ref.requires_grad_(False)

    def partial_diff_and_sample(self, x, c_img, c_txt, t=None, use_ref=False, mode="eval"):
        for i, step in enumerate(sorted(self.timesteps, reverse=True)):
            if mode == "eval":
                t = torch.full((x.size(0),), step, device=self.device, dtype=torch.long)

            if use_ref:
                eps = self.unet_ref(x, t, encoder_hidden_states=c_txt).sample
            else:
                eps = self.unet(x, t, encoder_hidden_states=c_txt).sample

            index = self.t2index(t)
            x0_hat = self.spaced_sampler._predict_xstart_from_eps(x, index, eps)
            x = x0_hat
            if mode == "train":
                break
        return x

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

            self.disc_params = list(filter(lambda p: p.requires_grad, self.disc.parameters()))
            self.opt_disc = optimizer(self.disc_params, **opt_config.kwargs)
            self.sch_disc = CosineAnnealingLR(self.opt_disc, T_max=self.cfg.train.train_steps, eta_min=1e-7)

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
        attrs = ["res_model", "vae", "text_encoder", "unet", "unet_ref",
                 "seg_model", "seg_hq_model", "disc", "val_dataloader"]
        if not self.args.eval_only:
            attrs += ["opt_gen", "sch_gen", "opt_seg_model", "sch_seg_model",
                      "opt_disc", "sch_disc", "train_dataloader"]

        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)

    def train(self):
        self.loss_records = dict(HLF=[], REG=[], SEG=[], ADV=[], FM=[], DISC=[], D_REAL=[], D_FAKE=[])
        self.on_training_start()

        while self.global_step < self.max_steps:
            pbar = self.make_pbar(total=len(self.train_dataloader))
            for batch in self.train_dataloader:
                self.prepare_batch_inputs(batch, transform=True)
                b = self.gt.size(0)
                bh = b // 2

                with self.accelerator.autocast():
                    self.seg_model.eval().requires_grad_(False)
                    self.unwrap_model(self.disc).eval().requires_grad_(False)

                    pre_res, z_pre_res, c_txt = self.prepare_condition()
                    t = self.timesteps_pt[torch.randint(0, len(self.timesteps), (b,), device=self.device)]

                    # reference samples for task-preserving regression loss
                    with torch.no_grad():
                        z_res_ref = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt, t,
                                                                 use_ref=True, mode="train")
                        res_ref = wavelet_reconstruction(self.decode_image(z_res_ref), pre_res)
                        _, feat_res_ref = self.seg_model(res_ref, return_feat=True)
                        _, feat_hq_res_ref = self.seg_hq_model(res_ref, return_feat=True)

                    z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt, t, mode="train")
                    res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    # hlf loss
                    _, feat_gt = self.seg_model(self.gt, return_feat=True)
                    _, feat_res = self.seg_model(res, return_feat=True)
                    with torch.no_grad():
                        _, feat_hq_gt = self.seg_hq_model(self.gt, return_feat=True)
                    _, feat_hq_res = self.seg_hq_model(res, return_feat=True)
                    loss_hlf = self.calculate_hlf_loss(feat_res, feat_gt, feat_hq_res, feat_hq_gt)
                    loss_hlf *= self.cfg.train.weight.hlf

                    # task-preserving regression loss
                    loss_reg = self.calculate_hlf_loss(feat_res, feat_res_ref, feat_hq_res, feat_hq_res_ref)
                    loss_reg *= self.cfg.train.weight.reg

                    # adv loss
                    w_adv = self.cfg.train.weight.adv if self.global_step > self.cfg.train.get("adv_start", 0) else 0.0
                    loss_adv = self.disc(res, for_G=True).mean() * w_adv

                self.opt_gen.zero_grad()
                self.accelerator.backward(loss_hlf + loss_reg + loss_adv)
                self.opt_gen.step()
                self.sch_gen.step()

                with self.accelerator.autocast():
                    self.seg_model.train().requires_grad_(True)

                    with torch.no_grad():
                        z_res = self.partial_diff_and_sample(z_pre_res, z_pre_res, c_txt)
                        res = wavelet_reconstruction(self.decode_image(z_res), pre_res)

                    pred, feat_mix = self.seg_model(torch.cat((res[:bh], self.gt[bh:]), dim=0), return_feat=True)
                    loss_seg = F.cross_entropy(pred["out"], self.mask, ignore_index=255) * self.cfg.train.weight.seg
                    loss_fm = F.l1_loss(feat_mix["C5"], feat_hq_gt["C5"]) * self.cfg.train.weight.fm

                self.opt_seg_model.zero_grad()
                self.accelerator.backward(loss_seg + loss_fm)
                self.opt_seg_model.step()
                self.sch_seg_model.step()

                with self.accelerator.autocast():
                    self.unwrap_model(self.disc).train().requires_grad_(True)

                    loss_disc_real, real_logits = self.disc(self.gt, for_real=True, return_logits=True)
                    loss_disc_fake, fake_logits = self.disc(res, for_real=False, return_logits=True)
                    loss_disc = loss_disc_real.mean() + loss_disc_fake.mean()
                    with torch.no_grad():
                        real_logits = torch.tensor([logit.mean() for logit in real_logits], device=self.device).mean()
                        fake_logits = torch.tensor([logit.mean() for logit in fake_logits], device=self.device).mean()

                self.opt_disc.zero_grad()
                self.accelerator.backward(loss_disc)
                self.opt_disc.step()
                self.sch_disc.step()

                self.global_step += 1
                self.loss_records["HLF"].append(loss_hlf.item())
                self.loss_records["REG"].append(loss_reg.item())
                self.loss_records["ADV"].append(loss_adv.item())
                self.loss_records["SEG"].append(loss_seg.item())
                self.loss_records["FM"].append(loss_fm.item())
                self.loss_records["DISC"].append(loss_disc.item())
                self.loss_records["D_REAL"].append(real_logits.item())
                self.loss_records["D_FAKE"].append(fake_logits.item())
                pbar.update(1)
                pbar.set_description(
                    f"Epoch: {self.epoch:04d}, Steps: {self.global_step:07d}, "
                    f"HLF: {loss_hlf.item():.3f}, REG: {loss_reg.item():.3f}, ADV: {loss_adv.item():.3f}, "
                    # f"SEG: {loss_seg.item():.3f}, FM: {loss_fm.item():.3f}"
                    f"D_REAL: {real_logits.item():.3f}, D_FAKE: {fake_logits.item():.3f}"
                )

                if self.global_step % self.cfg.train.log_every == 0 or (self.args.debug):
                    self.log_training_metrics()
                if self.global_step % self.cfg.train.image_every == 0 or (self.args.debug):
                    self.seg_model.eval()
                    with self.accelerator.autocast(), torch.no_grad():
                        pred = self.seg_model(res)
                    self.seg_model.train()
                    img_dict={"pre_restored": pre_res, "restored": res,
                              "restored_ref": res_ref, "pred": mask2rgb(pred["out"].argmax(1))}
                    self.log_images(img_dict, N=4)
                if self.global_step % self.cfg.train.ckpt_every == 0 or (self.args.debug):
                    self.save_checkpoints()
                if self.global_step % self.cfg.val.val_every == 0 or (self.args.debug):
                    self.evaluate()

                if self.global_step >= self.max_steps:
                    break

            pbar.close()
            self.epoch += 1

        self.save_checkpoints(last=True)
        self.accelerator.end_training()
