import os

import psutil


def get_resource_config(
    n_cores: int | None = None,
    memory_gb: float | None = None,
) -> tuple[int, float]:
    """
    Resolve the number of CPU cores and memory (GB) to use for a processing job.

    Priority order:
    1. Explicitly provided values
    2. SLURM environment variables (SLURM_CPUS_PER_TASK, SLURM_MEM_PER_NODE)
    3. psutil auto-detection: all available cores and 80% of available memory

    On non-SLURM systems, warns if the requested values exceed what is available.

    Parameters
    ----------
    n_cores : int or None
        Number of CPU cores to use. If None, auto-detected.
    memory_gb : float or None
        Memory budget in GB. If None, auto-detected.

    Returns
    -------
    n_cores : int
    memory_gb : float
    """
    slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    on_slurm = slurm_mem_mb is not None or slurm_cpus is not None

    if n_cores is None:
        if slurm_cpus is not None:
            n_cores = int(slurm_cpus)
        else:
            n_cores = os.cpu_count() or 1

    if memory_gb is None:
        if slurm_mem_mb is not None:
            memory_gb = int(slurm_mem_mb) / 1024.0
        else:
            memory_gb = psutil.virtual_memory().available / 1e9

    if not on_slurm:
        available_memory_gb = psutil.virtual_memory().available / 1e9
        available_cores = os.cpu_count() or 1
        if memory_gb > available_memory_gb:
            print(
                f"Warning: requested memory {memory_gb:.1f} GB exceeds "
                f"currently available {available_memory_gb:.1f} GB."
            )
        if n_cores > available_cores:
            print(
                f"Warning: requested {n_cores} cores exceeds "
                f"available {available_cores}."
            )

    return n_cores, memory_gb
