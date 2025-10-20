"""
Apple DMG (Disk Image) unpacker component for OFRAK.

This module provides support for unpacking Apple DMG (Disk Image) files, which are
commonly used for software distribution on macOS.

## DMG File Format

DMG files come in two main formats:

### Simple Compressed DMG (no plist)
When the koly block has `xml_length = 0`, the DMG contains:
1. **Compressed/Raw Data**: The entire disk image (possibly zlib/bzip2 compressed)
2. **Koly Block**: A 512-byte trailer with metadata

### Complex DMG (with plist metadata)
Modern DMG files use a more complex structure:
1. **Data Fork**: Contains the actual compressed/encoded disk data in chunks
2. **XML Property List**: Describes the block mapping and compression scheme
3. **Koly Block**: A 512-byte trailer with metadata and pointers

## Supported Compression Methods

- **Raw/Uncompressed (0x00000001)**: Direct data copy
- **Zero-fill (0x00000000)**: Zero-filled sectors for sparse data
- **zlib (0x80000005)**: DEFLATE compression
- **bzip2 (0x80000006)**: bzip2 compression
- **ADC (0x80000004)**: Apple Data Compression (not yet implemented)

## Usage Example

```python
from ofrak import OFRAK

# Create OFRAK context
ofrak = OFRAK()
root_resource = await ofrak.create_root_resource_from_file("example.dmg")

# Identify and unpack
await root_resource.identify()
await root_resource.unpack()

# Access extracted disk image
disk_image = await root_resource.get_only_child()
```

## Technical Details

The unpacker follows these steps:

1. Parse the koly block trailer (last 512 bytes) to locate metadata
2. Extract and parse the XML property list (if present)
3. Process block map (blkx) entries containing mish blocks
4. Decompress each chunk based on its compression type
5. Reconstruct the original disk image from chunks

All multi-byte values in the DMG format are stored in big-endian byte order.

## Limitations

- ADC (Apple Data Compression) is not currently supported and requires external tools
- LZFSE compression is not yet implemented
- Some complex DMG features (e.g., encrypted images) are not supported

## References

- DMG Format Specification: https://newosxbook.com/DMG.html
- Apple Developer Documentation: https://developer.apple.com/documentation/

Based on specification from https://newosxbook.com/DMG.html
"""

import base64
import bz2
import zlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ofrak.component.analyzer import Analyzer
from ofrak.component.identifier import Identifier
from ofrak.component.packer import Packer
from ofrak.component.unpacker import Unpacker
from ofrak.core.binary import GenericBinary
from ofrak.core.magic import MagicDescriptionPattern, MagicMimePattern
from ofrak.resource import Resource
from ofrak.resource_view import ResourceView
from ofrak_type.range import Range

try:
    import plistlib
except ImportError:
    plistlib = None


# Compression type constants
COMPRESSION_ZERO_FILL = 0x00000000
COMPRESSION_RAW = 0x00000001
COMPRESSION_ADC = 0x80000004
COMPRESSION_ZLIB = 0x80000005
COMPRESSION_BZIP2 = 0x80000006
COMPRESSION_COMMENT = 0x7FFFFFFE
COMPRESSION_TERMINATOR = 0xFFFFFFFF


@dataclass
class KolyBlock(ResourceView):
    """
    The koly block is a 512-byte trailer at the end of a DMG file.
    All fields are in big-endian byte order.

    This ResourceView represents the parsed koly block metadata.
    """

    signature: bytes  # Should be b'koly'
    version: int
    header_size: int
    flags: int
    running_data_fork_offset: int
    data_fork_offset: int
    data_fork_length: int
    rsrc_fork_offset: int
    rsrc_fork_length: int
    segment_number: int
    segment_count: int
    segment_id: bytes  # 16 bytes UUID
    data_checksum_type: int
    data_checksum_size: int
    data_checksum: bytes  # 32 bytes
    xml_offset: int
    xml_length: int
    reserved1: bytes  # 120 bytes
    master_checksum_type: int
    master_checksum_size: int
    master_checksum: bytes  # 32 bytes
    image_variant: int
    sector_count: int
    reserved2: int
    reserved3: int
    reserved4: int


@dataclass
class MishBlock(ResourceView):
    """
    Mish block describes the block mapping for a partition.
    Contains chunk entries that map logical sectors to compressed data.
    """

    signature: bytes  # Should be b'mish'
    version: int
    sector_number: int
    sector_count: int
    data_offset: int
    buffers_needed: int
    block_descriptors: int
    reserved: bytes
    checksum_type: int
    checksum_size: int
    checksum: bytes
    chunks: List["ChunkEntry"]


@dataclass
class ChunkEntry:
    """
    A single chunk entry in a mish block.
    Maps a logical sector range to compressed data in the file.

    Each entry is 40 bytes:
    - entry_type (4 bytes): Compression method
    - sector_number (8 bytes): Starting logical sector
    - sector_count (8 bytes): Number of sectors
    - compressed_offset (8 bytes): Offset in data fork
    - compressed_length (8 bytes): Compressed data length
    """

    entry_type: int  # Compression method
    sector_number: int
    sector_count: int
    compressed_offset: int
    compressed_length: int


@dataclass
class DmgImage(GenericBinary):
    """
    An Apple DMG (Disk Image) file.
    """

    async def get_disk_image(self) -> Resource:
        """Get the extracted disk image resource."""
        return await self.resource.get_only_child()


class KolyBlockAnalyzer(Analyzer[None, KolyBlock]):
    """
    Analyzer to deserialize the KolyBlock (512-byte trailer) from a DMG file.

    The koly block is always located at the last 512 bytes of a DMG file and
    contains all metadata about the image structure, compression, and checksums.
    """

    id = b"KolyBlockAnalyzer"
    targets = (KolyBlock,)
    outputs = (KolyBlock,)

    async def analyze(self, resource: Resource, config=None) -> KolyBlock:
        """
        Parse the koly block from the last 512 bytes of the DMG.

        Returns:
            KolyBlock with all fields deserialized

        Raises:
            ValueError: If signature is not 'koly' or size is not 512 bytes
        """
        import struct

        data = await resource.get_data()
        if len(data) != 512:
            raise ValueError(f"Koly block must be 512 bytes, got {len(data)}")

        # Parse koly block fields using struct.unpack (big-endian)
        # Total must be 512 bytes - current format is 326 bytes, so add 186 bytes padding
        (
            signature,
            version,
            header_size,
            flags,
            running_data_fork_offset,
            data_fork_offset,
            data_fork_length,
            rsrc_fork_offset,
            rsrc_fork_length,
            segment_number,
            segment_count,
            segment_id,
            reserved_6,
            data_checksum_type,
            data_checksum_size,
            data_checksum,
            xml_offset,
            xml_length,
            reserved1,
            master_checksum_type,
            master_checksum_size,
            master_checksum,
            image_variant,
            sector_count,
            reserved2,
            reserved3,
            reserved4,
            reserved_padding,
        ) = struct.unpack(">4sIIIQQQQQII16s6sII32sQQ120sII32sIQIII186s", data)

        if signature != b"koly":
            raise ValueError(f"Invalid koly signature: {signature}")

        return KolyBlock(
            signature=signature,
            version=version,
            header_size=header_size,
            flags=flags,
            running_data_fork_offset=running_data_fork_offset,
            data_fork_offset=data_fork_offset,
            data_fork_length=data_fork_length,
            rsrc_fork_offset=rsrc_fork_offset,
            rsrc_fork_length=rsrc_fork_length,
            segment_number=segment_number,
            segment_count=segment_count,
            segment_id=segment_id,
            data_checksum_type=data_checksum_type,
            data_checksum_size=data_checksum_size,
            data_checksum=data_checksum,
            xml_offset=xml_offset,
            xml_length=xml_length,
            reserved1=reserved1,
            master_checksum_type=master_checksum_type,
            master_checksum_size=master_checksum_size,
            master_checksum=master_checksum,
            image_variant=image_variant,
            sector_count=sector_count,
            reserved2=reserved2,
            reserved3=reserved3,
            reserved4=reserved4,
        )


class MishBlockAnalyzer(Analyzer[None, MishBlock]):
    """
    Analyzer to deserialize a MishBlock from its binary representation.

    Mish blocks are embedded in the XML plist as base64-encoded data and describe
    the block mapping for a partition, including compression information.
    """

    id = b"MishBlockAnalyzer"
    targets = (MishBlock,)
    outputs = (MishBlock,)

    async def analyze(self, resource: Resource, config=None) -> MishBlock:
        """
        Parse a mish block from its binary data.

        Returns:
            MishBlock with all fields and chunk entries deserialized

        Raises:
            ValueError: If signature is not 'mish'
        """
        import struct

        data = await resource.get_data()

        # Parse mish header (204 bytes total) using struct.unpack (big-endian)
        (
            signature,
            version,
            sector_number,
            sector_count,
            data_offset,
            buffers_needed,
            block_descriptors,
            reserved_48,
            checksum_type,
            checksum_size,
            checksum,
            reserved_76,
        ) = struct.unpack(">4sIQQQII48sII32s76s", data[:204])

        if signature != b"mish":
            raise ValueError(f"Invalid mish signature: {signature}")

        # Parse chunk entries (40 bytes each)
        chunks = []
        offset = 204
        for _ in range(block_descriptors):
            (
                entry_type,
                padding,
                sector_num,
                sec_count,
                comp_offset,
                comp_length,
            ) = struct.unpack(">I4sQQQQ", data[offset:offset+40])
            offset += 40

            chunks.append(
                ChunkEntry(
                    entry_type=entry_type,
                    sector_number=sector_num,
                    sector_count=sec_count,
                    compressed_offset=comp_offset,
                    compressed_length=comp_length,
                )
            )

        return MishBlock(
            signature=signature,
            version=version,
            sector_number=sector_number,
            sector_count=sector_count,
            data_offset=data_offset,
            buffers_needed=buffers_needed,
            block_descriptors=block_descriptors,
            reserved=reserved_48 + checksum + reserved_76,
            checksum_type=checksum_type,
            checksum_size=checksum_size,
            checksum=checksum,
            chunks=chunks,
        )


class KolyBlockIdentifier(Identifier):
    """
    Identifier for Apple DMG files based on the koly block signature.

    DMG files have a 512-byte koly block at the end of the file that starts
    with the signature b'koly'. This identifier checks for that signature
    since libmagic cannot reliably identify DMG files.
    """

    id = b"KolyBlockIdentifier"
    targets = (GenericBinary,)

    async def identify(self, resource: Resource, config=None) -> None:
        """
        Check if resource is a DMG file by looking for koly signature.

        Args:
            resource: The resource to identify
            config: Optional configuration (unused)
        """
        # DMG files must be at least 512 bytes (the koly block size)
        data = await resource.get_data()
        if len(data) < 512:
            return

        # Check for 'koly' signature in the last 512 bytes
        koly_block = data[-512:]
        if koly_block[:4] == b"koly":
            resource.add_tag(DmgImage)


class DmgUnpacker(Unpacker[None]):
    """
    Unpacker for Apple DMG (Disk Image) files.

    Supports various compression methods including:
    - Raw/uncompressed data
    - Zero-fill
    - zlib compression
    - bzip2 compression
    - ADC (Apple Data Compression) - requires external tool

    Handles both simple compressed DMGs (xml_length=0) and complex DMGs
    with XML plist metadata containing block descriptors.

    Based on specification from https://newosxbook.com/DMG.html
    """

    id = b"DmgUnpacker"
    targets = (DmgImage,)
    children = (GenericBinary, KolyBlock)
    external_dependencies = ()  # ADC support would require external tool

    async def unpack(self, resource: Resource, config=None) -> None:
        """
        Unpack a DMG file by:
        1. Creating a child resource for the koly block trailer
        2. If XML plist exists: parse blkx entries and decompress chunks
        3. If no XML plist: treat as simple compressed/raw disk image
        4. Reconstructing the disk image
        """
        import struct

        dmg_data = await resource.get_data()
        dmg_size = len(dmg_data)

        # Create child resource for koly block (last 512 bytes)
        # This will be automatically analyzed by KolyBlockAnalyzer
        koly_resource = await resource.create_child(
            tags=(KolyBlock,),
            data_range=Range(dmg_size - 512, dmg_size),
        )
        koly = await koly_resource.view_as(KolyBlock)

        # Check if this is a simple compressed DMG without plist metadata
        if koly.xml_length == 0:
            # Simple compressed DMG: data before koly block is the disk image
            # Create child resource and let OFRAK's existing compression unpackers
            # (ZlibUnpacker, Bzip2Unpacker, etc.) handle decompression automatically
            await resource.create_child(
                tags=(GenericBinary,),
                data_range=Range(0, dmg_size - 512),
            )
            return

        # Modern DMG with XML plist metadata
        # Extract XML property list
        xml_data = dmg_data[koly.xml_offset : koly.xml_offset + koly.xml_length]

        if plistlib is None:
            raise RuntimeError("plistlib is required to unpack DMG files with metadata")

        # Parse the plist
        plist = plistlib.loads(xml_data)

        # Find all blkx entries (block map descriptors)
        resource_fork = plist.get("resource-fork", {})
        blkx_entries = resource_fork.get("blkx", [])

        if not blkx_entries:
            raise ValueError("No blkx entries found in DMG plist")

        # Process each blkx entry and extract chunks
        all_chunks: List[Tuple[int, bytes]] = []

        for blkx in blkx_entries:
            # The 'Data' field contains base64-encoded mish block
            mish_data_b64 = blkx.get("Data")
            if not mish_data_b64:
                continue

            mish_data = base64.b64decode(mish_data_b64)

            # Parse mish block directly
            mish = self._parse_mish_block(mish_data)

            # Process each chunk in the mish block
            for chunk in mish.chunks:
                if chunk.entry_type == COMPRESSION_TERMINATOR:
                    # End of chunks
                    break

                decompressed_data = self._decompress_chunk(
                    dmg_data, chunk, koly.data_fork_offset
                )

                if decompressed_data:
                    # Store with sector position for proper ordering
                    all_chunks.append((chunk.sector_number, decompressed_data))

        # Sort chunks by sector number and concatenate
        all_chunks.sort(key=lambda x: x[0])
        disk_image_data = b"".join(chunk_data for _, chunk_data in all_chunks)

        # Create child resource with the extracted disk image
        await resource.create_child(
            tags=(GenericBinary,),
            data=disk_image_data,
        )

    def _decompress_chunk(
        self, dmg_data: bytes, chunk: ChunkEntry, data_fork_offset: int
    ) -> Optional[bytes]:
        """
        Decompress a single chunk based on its compression type.

        Args:
            dmg_data: Full DMG file data
            chunk: Chunk entry describing the compressed data
            data_fork_offset: Offset to the data fork in the DMG file

        Returns:
            Decompressed data or None if chunk should be skipped
        """
        chunk_type = chunk.entry_type

        # Zero-fill: return zeros
        if chunk_type == COMPRESSION_ZERO_FILL:
            return b"\x00" * (chunk.sector_count * 512)

        # Comment block: skip
        if chunk_type == COMPRESSION_COMMENT:
            return None

        # Terminator: skip
        if chunk_type == COMPRESSION_TERMINATOR:
            return None

        # Extract compressed data from file
        offset = data_fork_offset + chunk.compressed_offset
        compressed_data = dmg_data[offset : offset + chunk.compressed_length]

        # Raw/uncompressed data
        if chunk_type == COMPRESSION_RAW:
            return compressed_data

        # zlib compression
        if chunk_type == COMPRESSION_ZLIB:
            try:
                return zlib.decompress(compressed_data)
            except zlib.error as e:
                raise ValueError(f"Failed to decompress zlib chunk: {e}")

        # bzip2 compression
        if chunk_type == COMPRESSION_BZIP2:
            try:
                return bz2.decompress(compressed_data)
            except Exception as e:
                raise ValueError(f"Failed to decompress bzip2 chunk: {e}")

        # ADC (Apple Data Compression)
        if chunk_type == COMPRESSION_ADC:
            raise NotImplementedError(
                "ADC (Apple Data Compression) is not yet supported. "
                "This would require an external tool or library."
            )

        # Unknown compression type
        raise ValueError(f"Unknown compression type: 0x{chunk_type:08x}")

    @staticmethod
    def _parse_koly_block(data: bytes) -> KolyBlock:
        """
        Parse a koly block from raw bytes without creating a resource.

        Args:
            data: 512 bytes of koly block data

        Returns:
            KolyBlock with all fields deserialized

        Raises:
            ValueError: If signature is not 'koly' or size is not 512 bytes
        """
        import struct

        if len(data) != 512:
            raise ValueError(f"Koly block must be 512 bytes, got {len(data)}")

        # Parse koly block (512 bytes total) using struct.unpack (big-endian)
        (
            signature,
            version,
            header_size,
            flags,
            running_data_fork_offset,
            data_fork_offset,
            data_fork_length,
            rsrc_fork_offset,
            rsrc_fork_length,
            segment_number,
            segment_count,
            segment_id,
            reserved_6,  # 6 bytes padding - not stored in KolyBlock
            data_checksum_type,
            data_checksum_size,
            data_checksum,
            xml_offset,
            xml_length,
            reserved1,
            master_checksum_type,
            master_checksum_size,
            master_checksum,
            image_variant,
            sector_count,
            reserved2,
            reserved3,
            reserved4,
            reserved_padding,  # 186 bytes padding - not stored in KolyBlock
        ) = struct.unpack(">4sIIIQQQQQII16s6sII32sQQ120sII32sIQIII186s", data)

        if signature != b"koly":
            raise ValueError(f"Invalid koly signature: {signature}")

        return KolyBlock(
            signature=signature,
            version=version,
            header_size=header_size,
            flags=flags,
            running_data_fork_offset=running_data_fork_offset,
            data_fork_offset=data_fork_offset,
            data_fork_length=data_fork_length,
            rsrc_fork_offset=rsrc_fork_offset,
            rsrc_fork_length=rsrc_fork_length,
            segment_number=segment_number,
            segment_count=segment_count,
            segment_id=segment_id,
            data_checksum_type=data_checksum_type,
            data_checksum_size=data_checksum_size,
            data_checksum=data_checksum,
            xml_offset=xml_offset,
            xml_length=xml_length,
            reserved1=reserved1,
            master_checksum_type=master_checksum_type,
            master_checksum_size=master_checksum_size,
            master_checksum=master_checksum,
            image_variant=image_variant,
            sector_count=sector_count,
            reserved2=reserved2,
            reserved3=reserved3,
            reserved4=reserved4,
        )

    @staticmethod
    def _parse_mish_block(data: bytes) -> MishBlock:
        """
        Parse a mish block from raw bytes without creating a resource.

        Args:
            data: Raw bytes of mish block (204 bytes header + 40 bytes per chunk)

        Returns:
            MishBlock with all fields and chunk entries deserialized

        Raises:
            ValueError: If signature is not 'mish'
        """
        import struct

        # Parse mish header (204 bytes total) using struct.unpack (big-endian)
        (
            signature,
            version,
            sector_number,
            sector_count,
            data_offset,
            buffers_needed,
            block_descriptors,
            reserved_48,
            checksum_type,
            checksum_size,
            checksum,
            reserved_76,
        ) = struct.unpack(">4sIQQQII48sII32s76s", data[:204])

        if signature != b"mish":
            raise ValueError(f"Invalid mish signature: {signature}")

        # Parse chunk entries (40 bytes each)
        chunks = []
        offset = 204
        for _ in range(block_descriptors):
            (
                entry_type,
                padding,
                sector_num,
                sec_count,
                comp_offset,
                comp_length,
            ) = struct.unpack(">I4sQQQQ", data[offset : offset + 40])
            offset += 40

            chunks.append(
                ChunkEntry(
                    entry_type=entry_type,
                    sector_number=sector_num,
                    sector_count=sec_count,
                    compressed_offset=comp_offset,
                    compressed_length=comp_length,
                )
            )

        return MishBlock(
            signature=signature,
            version=version,
            sector_number=sector_number,
            sector_count=sector_count,
            data_offset=data_offset,
            buffers_needed=buffers_needed,
            block_descriptors=block_descriptors,
            reserved=reserved_48 + checksum + reserved_76,
            checksum_type=checksum_type,
            checksum_size=checksum_size,
            checksum=checksum,
            chunks=chunks,
        )


class DmgPacker(Packer[None]):
    """
    Packer for Apple DMG (Disk Image) files.

    Recreates a DMG file from an unpacked disk image by:
    1. Compressing the disk image data (zlib by default)
    2. Creating a new koly block with updated offsets and sizes
    3. Reconstructing the DMG file structure

    Note: Currently only supports simple compressed DMGs (xml_length=0).
    Complex DMGs with XML plist metadata are not yet supported for packing.
    """

    targets = (DmgImage,)

    async def pack(self, resource: Resource, config=None):
        """
        Pack a DMG file from its child disk image.

        Raises:
            NotImplementedError: For complex DMGs with plist metadata
        """
        dmg_view = await resource.view_as(DmgImage)

        # Parse the original koly block to check format
        original_data = await resource.get_data()
        original_koly = DmgUnpacker._parse_koly_block(original_data[-512:])

        # Get the disk image child
        disk_image = await dmg_view.get_disk_image()
        disk_data = await disk_image.get_data()

        # Only support simple compressed DMGs for now
        if original_koly.xml_length > 0:
            raise NotImplementedError(
                "Packing complex DMGs with plist metadata is not yet supported"
            )

        # Compress the disk image data
        compressed_data = zlib.compress(disk_data, level=9)

        # Create new koly block with updated metadata
        new_koly = self._create_koly_block(
            original_koly=original_koly,
            sector_count=len(disk_data) // 512,
        )

        # Reconstruct DMG file
        new_dmg_data = compressed_data + new_koly

        # Patch the resource with new data
        original_size = await resource.get_data_length()
        resource.queue_patch(Range(0, original_size), data=new_dmg_data)

    def _create_koly_block(self, original_koly: KolyBlock, sector_count: int) -> bytes:
        """
        Create a new koly block with updated metadata.

        Args:
            original_koly: The original koly block to base new one on
            sector_count: Number of 512-byte sectors in the disk image

        Returns:
            512 bytes of serialized koly block data
        """
        import struct

        # Serialize all koly block fields using struct.pack (big-endian)
        # Must be 512 bytes total - adding 186 bytes padding at end
        return struct.pack(
            ">4sIIIQQQQQII16s6sII32sQQ120sII32sIQIII186s",
            b"koly",
            original_koly.version,
            512,  # header_size
            original_koly.flags,
            0,  # running_data_fork_offset
            0,  # data_fork_offset
            0,  # data_fork_length (simple compressed)
            0,  # rsrc_fork_offset
            0,  # rsrc_fork_length
            original_koly.segment_number,
            original_koly.segment_count,
            original_koly.segment_id,
            b"\x00" * 6,  # reserved
            original_koly.data_checksum_type,
            original_koly.data_checksum_size,
            original_koly.data_checksum,
            0,  # xml_offset (no plist)
            0,  # xml_length (no plist)
            original_koly.reserved1,
            original_koly.master_checksum_type,
            original_koly.master_checksum_size,
            original_koly.master_checksum,
            original_koly.image_variant,
            sector_count,
            0,  # reserved2
            0,  # reserved3
            0,  # reserved4
            b"\x00" * 186,  # reserved padding to reach 512 bytes
        )


# Register DMG file type with magic patterns
MagicDescriptionPattern.register(
    DmgImage, lambda s: "Apple Disk Image" in s or "Apple Driver Map" in s
)

# Some DMG files may be detected as application/x-apple-diskimage
MagicMimePattern.register(DmgImage, "application/x-apple-diskimage")
