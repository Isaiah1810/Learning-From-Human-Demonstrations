from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DistributedType
from pathlib import Path
from beartype import beartype
from .action_predictor import VideoToAction


def cycle(dl, skipped_dl=None):
    """
    Helper function to wrap a dataloader into an infinite iterator,
    optionally yielding from a partially skipped dataloader first.
    """
    if skipped_dl is not None:
        for data in skipped_dl:
            yield data
    while True:
        for data in dl:
            yield data


@beartype
class VideoActionTrainer(nn.Module):
    def __init__(
            self,
            model: VideoToAction,
            dataset: torch.utils.data.Dataset,
            batch_size: int,
            num_train_steps: int,
            results_folder: str,
            lr: float = 3e-4,
            grad_accum_every: int = 1,
            max_grad_norm: float = 0.5,
            use_ema: Optional[bool] = False,
            save_model_every: int = 1000,
            save_milestone_every: int = 10000,
            accelerator_kwargs: dict = {},
            resume_checkpoint: Optional[str] = None,
            milestone_optim: bool = True,
            wandb_kwargs: dict = {}
        ):
            super().__init__()

            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(**accelerator_kwargs, kwargs_handlers=[ddp_kwargs])
            self.accelerator.init_trackers(
                project_name=wandb_kwargs.get("project", "video-action"),
                config=wandb_kwargs.get("config"),
                init_kwargs={"wandb": wandb_kwargs}
            )

            # prepare everything together
            self.model = model

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(parents=True, exist_ok=True)

            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

            self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
                self.model.to(self.accelerator.device), self.optimizer, self.dataloader)

            self.grad_accum_every = grad_accum_every
            self.max_grad_norm = max_grad_norm
            self.save_model_every = save_model_every
            self.save_milestone_every = save_milestone_every
            self.milestone_optim = milestone_optim

            self.num_train_steps = num_train_steps
            self.steps = 0
            self.resume_checkpoint = resume_checkpoint

            if resume_checkpoint is not None and Path(resume_checkpoint).exists():
                self.load(resume_checkpoint)
                num_batches_to_skip = self.steps % len(self.dataloader)
                skipped_dl = self.accelerator.skip_first_batches(self.dataloader, num_batches_to_skip)
                self.dl_iter = cycle(self.dataloader, skipped_dl)
                self.accelerator.print(f"Resumed from checkpoint {resume_checkpoint} at step {self.steps}")
            else:
                self.dl_iter = cycle(self.dataloader)

            self.accelerator.wait_for_everyone()

    # Accelerator helpers
    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process


    def get_dl_state(self, data_iter):
        return getattr(data_iter, 'state', None)

    def load(self, path):
        path = Path(path)
        if not path.exists():
            self.print(f"Checkpoint not found at {path}")
            return

        state = torch.load(path, map_location='cpu')
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optim'])
        self.steps = int(state['steps'])  # make sure it's a scalar int
        self.print(f"Resumed from checkpoint at step {self.steps}")

    def save(self, path, milestone=False, milestone_optim=True):
        state = {
            'model': self.accelerator.get_state_dict(self.model),
            'steps': self.steps
        }
        if not milestone or (milestone and milestone_optim):
            state['optim'] = self.optimizer.state_dict()

        torch.save(state, path)

    def train_step(self):
        self.model.train()
        total_loss = 0.0

        for _ in range(self.grad_accum_every):
            V, S, A, mask_V, mask_S = next(self.dl_iter)
            V, S, A, mask_V, mask_S = map(lambda x: x.to(self.device), (V, S, A, mask_V, mask_S))

            A_hat, loss = self.model(V, S, A, temporal_mask_V=mask_V, temporal_mask_S=mask_S, context_mask=mask_V)
            self.accelerator.backward(loss / self.grad_accum_every)
            total_loss += loss.item() / self.grad_accum_every

        if self.max_grad_norm:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        if self.is_main:
            # Compute parameter and gradient norm before zeroing
            param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in self.model.parameters() if p.requires_grad]))
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]))

            self.accelerator.log({
                "loss": total_loss,
                "step": self.steps,
                "param_norm": param_norm.item(),
                "grad_norm": grad_norm.item()
            })

        self.optimizer.zero_grad()
        return total_loss

    def train(self):
        while self.steps < self.num_train_steps:
            loss = self.train_step()

            if self.is_main:
                self.print(f"Step {self.steps}: Loss = {loss:.4f}")

            if self.is_main:
                if self.steps % self.save_model_every == 0:
                    self.save(self.results_folder / "current_model.pt")
                if self.steps % self.save_milestone_every == 0:
                    self.save(self.results_folder / f"model.{self.steps}.pt", milestone=True, milestone_optim=self.milestone_optim)

            self.steps += 1

        self.print("Training complete")
