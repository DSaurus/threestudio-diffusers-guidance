import threestudio
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from threestudio.utils.typing import *


def lcm_set_timesteps(self, *args, **kwargs):
    super(type(self), self).set_timesteps(*args, **kwargs)

    strength = self.lcm_timesteps.item() / self.config.num_train_timesteps
    init_timestep = min(
        int(self.num_inference_steps * strength), self.num_inference_steps
    )
    t_start = min(
        max(self.num_inference_steps - init_timestep - self.more_steps, 0),
        self.num_inference_steps - 1,
    )
    self.timesteps = self.timesteps[t_start:]
    self.num_inference_steps = len(self.timesteps)


@threestudio.register("lcm-sampler")
class LCMSampler:
    init_lcm: bool = False

    def init_lcm_sampler(
        self,
    ):
        LCMScheduler = type(
            "LCMScheduler",
            (type(self.pipe.scheduler),),
            {"set_timesteps": lcm_set_timesteps},
        )
        self.lcm_scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.lcm_scheduler.more_steps = 0
        self.init_lcm = True

    def compute_grad_lcm(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_lcm:
            self.init_lcm_sampler()
        batch_size = latents.shape[0]

        timesteps = t[:1]
        t[:] = t[:1]
        self.lcm_scheduler.lcm_timesteps = timesteps

        noise = torch.randn_like(latents)
        noisy_latents = self.lcm_scheduler.add_noise(latents, noise, t)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents
            self.pipe.scheduler = self.lcm_scheduler
            pred_latents = self.pipe(**kwargs).images
            if type(pred_latents) != torch.Tensor:
                pred_images = torch.from_numpy(pred_latents)
                pred_latents = self.prepare_latents(pred_images)
            # img = self.decode_latents(pred_latents)
            # import cv2
            # cv2.imwrite(".threestudio_cache/test.jpg", (img.permute(0, 2, 3, 1)[0].detach().cpu().numpy()[:, :, ::-1]*255))
            # exit(0)

        loss_lcm = (
            0.5
            * F.mse_loss(
                latents, pred_latents.clone().to(noisy_latents.device), reduction="sum"
            )
            / batch_size
        )

        return loss_lcm
