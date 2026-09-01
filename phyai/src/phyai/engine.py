"""Process-level engine for registered model plugins."""

from __future__ import annotations

import abc
import time
import threading
from typing import Any, ClassVar
from datetime import timedelta
from contextlib import contextmanager
from dataclasses import replace, dataclass
from collections.abc import Mapping, Iterable, Generator

import torch
import torch.distributed as dist
from torch import nn

import phyai.parallel as P
from phyai.utils import get_logger
from phyai.utils.cuda import init_cuda, format_gib, init_cublas, available_memory_bytes
from phyai.kernel.call import freeze_kernel_choices
from phyai.kernel.types import ModelContext
from phyai.engine_config import EngineConfig, init_engine_config
from phyai.parallel.dist import init_dist
from phyai.utils.logging import configure_logging
from phyai.utils.env_setup import init_env, set_ulimit, init_process_debug
from phyai.kernel.bootstrap import reset_kernel_selector, initialize_kernel_system
from phyai.utils.torch_setup import init_seed, disable_grad, init_threads
from phyai.runtime.tensor_dump import (
    TensorDumper,
    load_filter_fn,
    register_tensor_dumper,
)
from phyai.layers.attention.utils import release_global_fi_workspaces

logger = get_logger(__name__)


def _force_eager_for_dump(cfg: EngineConfig) -> EngineConfig:
    """Disable CUDA graphs in ``cfg`` when tensor dumping is enabled."""
    if not cfg.runtime.use_cuda_graph:
        return cfg
    return cfg.replace(runtime=replace(cfg.runtime, use_cuda_graph=False))


def _force_eager_without_cuda(cfg: EngineConfig) -> EngineConfig:
    """Disable CUDA graphs in ``cfg`` when the target is not CUDA."""
    if not cfg.runtime.use_cuda_graph:
        return cfg
    if torch.device(cfg.device.target).type == "cuda":
        return cfg
    return cfg.replace(runtime=replace(cfg.runtime, use_cuda_graph=False))


@dataclass
class EntryArgs:
    """Base class for plugin argument dataclasses."""


class Entry(abc.ABC):
    """Interface for a model plugin's setup, inference, and cleanup."""

    name: ClassVar[str]
    args_cls: ClassVar[type[EntryArgs]]

    @abc.abstractmethod
    def setup(self, args: EntryArgs) -> None:
        """Build the model, load weights, prepare runners / scheduler."""

    @abc.abstractmethod
    def step(self, request: Any) -> Any:
        """Run one inference round. Request / response shape is plugin-defined."""

    def rollout_step(self, request: Any) -> Any:
        """Run one RL rollout round when the plugin supports trajectory output."""
        raise NotImplementedError(
            f"plugin {self.name!r} does not implement rollout_step()."
        )

    def close(self) -> None:
        """Release pinned GPU resources. Default: no-op."""
        return

    def begin_weight_update(self) -> None:
        """Begin one streamed in-memory weight update.

        Plugins that support hot updates override this method together with
        :meth:`update_weights` and :meth:`finish_weight_update`.
        """
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Apply one named-tensor batch to the active update session."""
        del weights
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def finish_weight_update(self) -> Any:
        """Finalize the active update and refresh derived runtime state."""
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def abort_weight_update(self) -> None:
        """Discard bookkeeping for an interrupted update session."""

    def dump_targets(self) -> dict[str, nn.Module]:
        """Return the root modules that should be included in tensor dumps."""
        return {}


@dataclass
class EngineArgs:
    """Select a plugin and provide its arguments and optional config."""

    plugin: str
    plugin_args: EntryArgs
    config: EngineConfig | None = None


class Engine:
    """In-process dispatcher for registered model plugins."""

    _plugins: ClassVar[dict[str, type[Entry]]] = {}

    @classmethod
    def register(cls, entry_cls: type[Entry]) -> type[Entry]:
        """Register a plugin entry class and return it unchanged."""
        if not isinstance(entry_cls, type) or not issubclass(entry_cls, Entry):
            raise TypeError(
                f"Engine.register expected an Entry subclass, got {entry_cls!r}."
            )
        name = getattr(entry_cls, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"{entry_cls.__name__}.name must be a non-empty string.")
        args_cls = getattr(entry_cls, "args_cls", None)
        if not isinstance(args_cls, type) or not issubclass(args_cls, EntryArgs):
            raise TypeError(
                f"{entry_cls.__name__}.args_cls must be an EntryArgs subclass."
            )
        existing = cls._plugins.get(name)
        if existing is not None and existing is not entry_cls:
            raise ValueError(
                f"plugin name {name!r} is already registered to {existing.__name__}."
            )
        cls._plugins[name] = entry_cls
        return entry_cls

    @classmethod
    def registered(cls) -> tuple[str, ...]:
        """Return all registered plugin names in registration order."""
        return tuple(cls._plugins.keys())

    def __init__(self, args: EngineArgs) -> None:
        # 1. Resolve the effective engine configuration.
        resolved = EngineConfig.from_env(base=args.config)
        self._dump_enabled = resolved.runtime.debug_tensor_dump_dir is not None
        if self._dump_enabled:
            forced = _force_eager_for_dump(resolved)
            if forced is not resolved:
                logger.warning_rank0(
                    "Tensor dump enabled (debug_tensor_dump_dir=%s): forcing "
                    "use_cuda_graph=False. Forward hooks cannot fire during a captured "
                    "CUDA-graph replay, so activation capture runs eager-only (slower "
                    "than the normal graph path).",
                    resolved.runtime.debug_tensor_dump_dir,
                )
            resolved = forced
        resolved = _force_eager_without_cuda(resolved)

        device_type = torch.device(resolved.device.target).type

        # 2. Initialize logging and the process environment.
        configure_logging()
        init_env(world_size=resolved.parallel.world_size, device_type=device_type)
        set_ulimit()
        init_process_debug()

        self.config: EngineConfig = init_engine_config(resolved)
        self._dumper: TensorDumper | None = None
        self._t_start = time.perf_counter()

        # 3. Initialize PyTorch process state.
        with self._stage("torch_setup"):
            init_threads(
                device_type=device_type, num_threads=self.config.runtime.num_threads
            )
            init_seed(self.config.runtime.seed)
            disable_grad()

        # 4. Initialize CUDA and cuBLAS.
        with self._stage("cuda"):
            init_cuda(self.config.device.target, self.config.device.params_dtype)
            init_cublas()

        parallel = self.config.parallel

        # 5. Initialize the distributed process group.
        with self._stage("dist"):
            self._owns_pg: bool = init_dist(
                world_size=parallel.world_size,
                device_type=device_type,
                timeout=timedelta(seconds=self.config.runtime.dist_timeout_s),
            )

        # 6. Initialize the parallel mesh and warm its communicators.
        with self._stage("mesh"):
            mesh = P.init(
                layout=(
                    parallel.dp_size,
                    parallel.cfg_size,
                    parallel.ep_size,
                    parallel.sp_size,
                    parallel.cp_size,
                    parallel.tp_size,
                ),
                mesh_dim_names=("dp", "cfg", "ep", "sp", "cp", "tp"),
                device=device_type,
            )

        process_title = f"phyai::{args.plugin}"
        for axis in ("dp", "tp"):
            if mesh.axis_size(axis) > 1:
                process_title += f"_{axis.upper()}{mesh.axis_local_rank(axis)}"
        init_process_debug(title=process_title)

        if parallel.world_size > 1:
            # Create communicators before any graph capture.
            with self._stage("collectives_warmup"):
                warmed = P.warmup_collectives()
            if warmed:
                logger.info_rank0("warmed collectives on axes %s", warmed)

        # 7. Initialize kernel selection.
        with self._stage("kernel"):
            self.kernel_selector = initialize_kernel_system(
                self.config.kernel,
                device=self.config.device.target,
                model=ModelContext(family=args.plugin),
            )

        # 8. Resolve and initialize the selected model plugin.
        entry_cls = self._plugins.get(args.plugin)
        if entry_cls is None:
            raise ValueError(
                f"unknown plugin {args.plugin!r}; registered: {list(self._plugins)!r}."
            )
        if not isinstance(args.plugin_args, entry_cls.args_cls):
            raise TypeError(
                f"plugin {entry_cls.name!r} expects "
                f"{entry_cls.args_cls.__name__}; got "
                f"{type(args.plugin_args).__name__}."
            )

        self.args = args
        self.entry: Entry = entry_cls()
        with self._stage("plugin_setup"):
            self.entry.setup(args.plugin_args)

        self._model_lock = threading.Lock()
        self._weight_update_active = False
        self._weight_update_received = False
        self._weight_update_failed = False
        self._version = 0

        # 9. Finalize kernel choices and attach debugging hooks.
        if self.config.runtime.freeze_kernel_choices:
            freeze_kernel_choices()

        if self._dump_enabled:
            self._dumper = self._build_dumper()

        free_now = available_memory_bytes(self.config.device.target)
        logger.info_rank0(
            "Engine ready (plugin=%s). total=%.2fs free=%s GiB",
            args.plugin,
            time.perf_counter() - self._t_start,
            format_gib(free_now),
        )

    @contextmanager
    def _stage(self, name: str) -> Generator[None]:
        """Time a bootstrap stage and log its device-memory change."""
        free_before = available_memory_bytes(self.config.device.target)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            free_after = available_memory_bytes(self.config.device.target)
            logger.info_rank0(
                "init stage=%s elapsed=%.2fs used=%s GiB free=%s GiB",
                name,
                elapsed,
                format_gib(max(0, free_before - free_after)),
                format_gib(free_after),
            )

    def _build_dumper(self) -> TensorDumper | None:
        """Build a tensor dumper from the plugin's dump targets."""
        runtime = self.config.runtime
        targets = self.entry.dump_targets()
        if not targets:
            logger.warning_rank0(
                "Tensor dump is enabled but plugin %r exposes no dump_targets(); "
                "nothing will be recorded. Override Entry.dump_targets() to return "
                "the model module(s) to capture.",
                self.args.plugin,
            )
            return None
        filter_spec = self._resolve_dump_filter()
        return register_tensor_dumper(
            targets,
            dump_dir=runtime.debug_tensor_dump_dir,
            filter=filter_spec,
        )

    def _resolve_dump_filter(self):
        """Resolve the configured tensor-dump filter."""
        runtime = self.config.runtime
        if runtime.debug_tensor_dump_filter_fn is not None:
            return load_filter_fn(runtime.debug_tensor_dump_filter_fn)
        return runtime.debug_tensor_dump_filter

    def step(self, request: Any) -> Any:
        """Run one inference round; forwards to the registered entry.

        When tensor dumping is active, the activations recorded during
        this round are flushed to a single ``pass{N}.pt`` file once the
        entry returns.
        """
        with self._model_lock:
            self._require_usable()
            result = self.entry.step(request)
            if self._dumper is not None:
                self._dumper.flush_pass()
            return result

    def rollout_step(self, request: Any) -> Any:
        """Run one RL rollout round through the registered entry."""
        with self._model_lock:
            self._require_usable()
            result = self.entry.rollout_step(request)
            if self._dumper is not None:
                self._dumper.flush_pass()
            return result

    @property
    def version(self) -> int:
        """Last fully committed in-memory weight version."""
        return self._version

    def _require_usable(self) -> None:
        if self._weight_update_failed:
            raise RuntimeError(
                "Engine is unavailable after a partially applied weight update; "
                "recreate it before further inference."
            )

    def begin_weight_update(self) -> None:
        """Begin one streamed hot update on the active plugin."""
        self._model_lock.acquire()
        try:
            self._require_usable()
            if self._weight_update_active:
                raise RuntimeError("A weight update is already active.")
            self.entry.begin_weight_update()
            self._weight_update_active = True
            self._weight_update_received = False
        except Exception:
            self._model_lock.release()
            raise

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Forward one named-tensor batch to the active plugin."""
        if not self._weight_update_active:
            raise RuntimeError("begin_weight_update() must be called first.")
        self._weight_update_received = True
        self.entry.update_weights(weights)

    def finish_weight_update(self, version: int | None = None) -> Any:
        """Finalize an update and commit its version only after validation."""
        if not self._weight_update_active:
            raise RuntimeError("No weight update is active.")
        try:
            report = self.entry.finish_weight_update()
            self._version = self._version + 1 if version is None else int(version)
            return report
        except Exception:
            if self._weight_update_received:
                self._weight_update_failed = True
            raise
        finally:
            self._weight_update_active = False
            self._weight_update_received = False
            self._model_lock.release()

    def abort_weight_update(self) -> None:
        """Abort bookkeeping for the active plugin update."""
        if not self._weight_update_active:
            return
        try:
            self.entry.abort_weight_update()
            if self._weight_update_received:
                self._weight_update_failed = True
        finally:
            self._weight_update_active = False
            self._weight_update_received = False
            self._model_lock.release()

    def close(self) -> None:
        """Release the plugin entry's resources, then tear down distributed
        state if the engine was the one to bring it up."""
        if self._weight_update_active:
            self.abort_weight_update()
        if self._dumper is not None:
            self._dumper.detach()
            self._dumper = None
        self.entry.close()
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_pg = False
        reset_kernel_selector()
        release_global_fi_workspaces()


__all__ = [
    "Engine",
    "EngineArgs",
    "Entry",
    "EntryArgs",
]


# Import plugin modules after defining Engine so their decorators register
# each Entry subclass. Add new plugin imports here.

from phyai.models.pi0 import main_pi0 as _main_pi0  # noqa: F401
from phyai.models.pi05 import main_pi05 as _main_pi05  # noqa: F401
from phyai.models.pi05 import main_pi05_wn as _main_pi05_wn  # noqa: F401
from phyai.models.cosmos3 import main_cosmos3 as _main_cosmos3  # noqa: F401
from phyai.models.cosmos3 import main_cosmos3_wn as _main_cosmos3_wn  # noqa: F401
from phyai.models.cosmos3 import (  # noqa: F401
    main_cosmos3_policy as _main_cosmos3_policy,
)
from phyai.models.cosmos3 import (  # noqa: F401
    main_cosmos3_policy_wn as _main_cosmos3_policy_wn,
)
from phyai.models.gr00t_n17 import main_gr00t_n17 as _main_gr00t_n17  # noqa: F401
from phyai.models.minicpm_gr00t import (  # noqa: F401
    main_minicpm_gr00t as _main_minicpm_gr00t,
)
