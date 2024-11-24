from typing import List


class KeepImageChannels:
    def __init__(self, channels: List[int]) -> None:
        """
        Initialize the transform with the given channels.
        Args:
            channels (List[int]): A list of integers representing the channels to be removed,
                      where 0: r, 1: g, 2: b, 3: a.
        """
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels, :, :]
