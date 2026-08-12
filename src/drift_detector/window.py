from collections import deque
from typing import Iterable, List


class SlidingWindow:
    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)

    def push(self, labels: Iterable[str]) -> None:
        self.buffer.append(list(labels))

    def is_full(self) -> bool:
        return len(self.buffer) == self.size

    def snapshot(self) -> List[List[str]]:
        return list(self.buffer)
