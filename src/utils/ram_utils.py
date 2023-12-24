"""Module provides functionality for monitoring and logging RAM usage of the current process and the overall system."""
import asyncio
import os
from typing import Tuple

import psutil

from src.utils.metrics import TOTAL_USED_RAM, USED_RAM

RAM_CHECK_TIMEOUT: int = 120


def process_memory_bytes() -> Tuple[int, int]:
    """
    Get the memory usage in bytes of the current process.

    Returns:
        tuple[int, int]: A tuple containing the memory usage in bytes (rss) of the current process and its PID.
    """
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    return mem_info.rss, pid


async def log_ram_memory(sleep_sec: int = RAM_CHECK_TIMEOUT) -> None:
    """
    Continuously log the RAM usage of the current process and the system.

    Args:
        sleep_sec (int): The interval in seconds between logging memory usage. Defaults to RAM_CHECK_TIMEOUT.
    """
    while True:
        bytes_ram_process, _ = process_memory_bytes()
        bytes_ram_total_used = psutil.virtual_memory().used
        TOTAL_USED_RAM.set(bytes_ram_total_used)
        USED_RAM.set(bytes_ram_process)
        await asyncio.sleep(sleep_sec)
