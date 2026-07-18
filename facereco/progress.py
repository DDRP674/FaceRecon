from __future__ import annotations

import sys


class SimpleProgress:
    def __init__(self, total: int, desc: str, every: int = 100):
        self.total = total
        self.desc = desc
        self.every = every
        self.n = 0
        print(f"{self.desc}: 0/{self.total}", file=sys.stderr, flush=True)

    def update(self, amount: int) -> None:
        self.n += amount
        if self.n >= self.total or self.n % self.every == 0:
            print(f"{self.desc}: {self.n}/{self.total}", file=sys.stderr, flush=True)

    def close(self) -> None:
        print(f"{self.desc}: {self.n}/{self.total} done", file=sys.stderr, flush=True)


def progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return SimpleProgress(total=total, desc=desc)
    return tqdm(total=total, desc=desc, dynamic_ncols=True)

