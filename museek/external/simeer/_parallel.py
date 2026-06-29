"""
Thin parallelism wrapper used by the TOD-sample loop.

The hot loop in :mod:`simeer.sky_integrator` is embarrassingly parallel
over time samples. We default to :mod:`joblib` with the Loky process
backend because:

*   the per-sample workload is numpy-heavy and not consistently
    GIL-free, so a thread pool would be CPU-bound on the GIL;
*   joblib's Loky backend works on a single node without extra launchers
    (unlike mpi4py, which requires ``mpiexec``);
*   joblib will fall back to serial execution when ``n_jobs == 1``,
    which keeps debugging straightforward.

A future :func:`map_samples_mpi` backend can be added without touching
the callers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, TypeVar

T = TypeVar("T")


def map_samples(
    func: Callable[..., T],
    iterable: Iterable[tuple],
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
    progress: bool = False,
    progress_desc: str = "TOD batches",
) -> list[T]:
    """Map ``func`` over an iterable of argument tuples.

    Parameters
    ----------
    func : callable
        Function applied to each tuple of arguments.
    iterable : iterable of tuples
        Argument tuples, one per task. (Each task may process more than
        one TOD sample when the caller has bundled them into batches.)
    n_jobs : int, default 1
        Number of worker processes. ``1`` runs serially without spawning
        joblib workers, which is the right behaviour for debugging.
        ``-1`` means "use all cores".
    backend : str, default ``'loky'``
        joblib backend. Use ``'threading'`` only when the worker is known
        to release the GIL throughout (rare for numpy-heavy work).
    verbose : int, default 0
        joblib verbosity level.
    progress : bool, default False
        If True, wrap the execution loop in :mod:`tqdm`. The bar
        increments as tasks complete, not as items are dispatched.
    progress_desc : str, default ``'TOD batches'``
        Label shown next to the progress bar.

    Returns
    -------
    list
        Results in input order.
    """
    items = list(iterable)

    if n_jobs == 1:
        if progress:
            from tqdm import tqdm

            return [func(*args) for args in tqdm(items, desc=progress_desc)]
        return [func(*args) for args in items]

    from joblib import Parallel, delayed  # type: ignore[import-not-found]

    # Wrap the *generator* of delayed calls in tqdm so the bar advances
    # as joblib dispatches and as results come back. joblib's own
    # ``verbose`` flag prints to stderr; tqdm gives a clean line.
    if progress:
        from tqdm import tqdm

        delayed_calls = (delayed(func)(*args) for args in tqdm(items, desc=progress_desc))
    else:
        delayed_calls = (delayed(func)(*args) for args in items)

    # ``max_nbytes='100K'`` lowers joblib's auto-memmap threshold (default
    # 1 MB) so the beam power cube -- which can be a few hundred KB for
    # synthetic test sizes and tens of MB for real MeerKLASS data -- is
    # always materialised on disk and memmapped into each worker,
    # avoiding redundant per-batch pickling.
    return Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes="100K",
    )(delayed_calls)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Translate a user-supplied ``n_jobs`` value to a positive integer.

    ``None`` -> 1; negative values are passed through to joblib (which
    interprets ``-1`` as "all cores").
    """
    if n_jobs is None:
        return 1
    return int(n_jobs)
