from dataclasses import dataclass
from enum import Enum
from ofrak.core.addressable import Addressable
from ofrak.model.resource_model import index


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

    @index
    def RefAddress(self) -> int:
        return self.ref_address

    @index
    def Direction(self) -> str:
        return self.direction.value

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
