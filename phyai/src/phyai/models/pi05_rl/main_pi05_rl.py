"""Opt-in pi0.5 plugin aligned with the RLinf actor's numeric boundaries."""

from __future__ import annotations

from typing import ClassVar

from phyai.engine import Engine, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args, PI05Entry, _compose_remap
from phyai.models.pi05_rl.modeling_pi05_rl import PI05RLModel
from phyai.models.pi05_rl.scheduler_ws1_pi05_rl import PI05RLWS1Scheduler
from phyai.utils import load_config
from phyai.weights import load_pretrained


@Engine.register
class PI05RLEntry(PI05Entry):
    """pi0.5 entry that opts into RLinf actor numerical alignment."""

    name: ClassVar[str] = "pi05_rl"
    args_cls: ClassVar[type[EntryArgs]] = PI05Args

    def setup(self, args: PI05Args) -> None:  # type: ignore[override]
        eng = get_engine_config()
        self.weight_remap = args.weight_remap
        self.require_full_hot_update = args.require_full_hot_update

        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, PI05Config)
        else:
            config = PI05Config()

        eng = self._apply_recommended_engine(eng, config)
        with use_quant_plan(load_quant_plan(args.checkpoint_dir)):
            self.model = PI05RLModel(
                config,
                vision_params_dtype=args.vision_params_dtype,
                add_value_head=args.add_value_head,
                device=eng.device.target,
            )

        if args.checkpoint_dir is not None:
            load_pretrained(
                self.model,
                args.checkpoint_dir,
                remap=_compose_remap(args.weight_remap),
                strict=args.weight_strict,
            )

        if self.model.value_head is not None:
            self.model.value_head.require_hot_update_weights()

        self.scheduler = PI05RLWS1Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            num_images=self._resolve_num_images(args.inputs_image_shape, config),
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler.setup()


__all__ = ["PI05RLEntry"]
