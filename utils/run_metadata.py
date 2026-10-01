"""Random-seed handling and reproducibility records for executed runs."""
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Do not silently change deterministic-kernel settings of historical runs.


def record_run(args, setting, seed):
    directory = Path('results') / setting
    directory.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = dict(arguments=vars(args), effective_seed=seed, git_commit=commit,
                  python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  split_boundary=os.environ.get('QAR_SPLIT_BOUNDARY', 'legacy_float'))
    (directory / 'run_config.json').write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
