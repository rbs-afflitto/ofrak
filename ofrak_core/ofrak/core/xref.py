from dataclasses import dataclass
from enum import Enum
from ofrak.core.addressable import Addressable


class XrefDirection(Enum):
    TO = "to"
    FROM = "from"


@dataclass
class Xref(Addressable):
    """
    A cross-reference to another virtual address.
    """

    ref_address: int
    direction: XrefDirection

    def __str__(self) -> str:
        if self.direction == XrefDirection.TO:
            arrow = "->"
        elif self.direction == XrefDirection.FROM:
            arrow = "<-"
        else:
            raise ValueError(f"Invalid direction value: {self.direction}")

        return f"Xref({hex(self.virtual_address)} {arrow} {hex(self.ref_address)})"

    def __repr__(self) -> str:
        return f"Xref({hex(self.virtual_address)}, {hex(self.ref_address)}, XrefDirection.{self.direction.name})"
