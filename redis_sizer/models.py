from dataclasses import dataclass
from enum import Enum


class MemoryUnit(str, Enum):
    """
    For displaying memory usage in different units.
    """

    B = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"

    def get_factor(self) -> int:
        """
        Get the conversion factor for the memory unit.
        """
        if self == MemoryUnit.B:
            return 1
        elif self == MemoryUnit.KB:
            return 1024
        elif self == MemoryUnit.MB:
            return 1024 * 1024
        elif self == MemoryUnit.GB:
            return 1024 * 1024 * 1024
        else:
            raise ValueError("Invalid value for memory_unit. Use B, KB, MB, or GB.")


@dataclass
class TableRow:
    """
    Represent a row in the result table.
    """

    key: str
    count: str
    size: str
    avg_size: str
    min_size: str
    max_size: str
    percentage: str
    level: int = 0  # Add level for indentation
    is_truncation_row: bool = False  # dimmed if True


@dataclass
class KeyNode:
    """
    Represent a node in the key hierarchy tree.
    """

    name: str  # The display name for this node
    full_path: str  # The full path from root to this node
    level: int  # The depth level in the hierarchy (0 for root, 1 for first level, etc.)
    keys: list[str]  # List of actual Redis keys that belong directly to this node
    size: int  # Total memory size of all keys in this subtree (including children)
    sizes: list[int]  # List of individual memory sizes for statistics (e.g., min, max, avg)
    children: dict[str, "KeyNode"]  # Child nodes in the hierarchy, keyed by their name
