from os import PathLike
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union

import accelerate
import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from beartype import beartype
from lion_pytorch import Lion
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader, random_split
from torch_optimizer import (
    PID,
    QHM,
    SGDP,
    SGDW,
    SWATS,
    AccSGD,
    AdaBound,
    AdaMod,
    AdamP,
    AggMo,
    DiffGrad,
    Lamb,
    NovoGrad,
    QHAdam,
    RAdam,
    Shampoo,
    Yogi,
)

from transformers.optimization import Adafactor

try:
    from accelerate.data_loader import MpDeviceLoaderWrapper
except ImportError:
    MpDeviceLoaderWrapper = DataLoader
    pass

try:
    from bitsandbytes.optim import Adam8bit, AdamW8bit, Lion8bit
except ImportError:
    Adam8bit = AdamW8bit = Lion8bit = None