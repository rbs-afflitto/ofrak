"""
Tests for Apple DMG unpacker component.
"""

import base64
import os
import plistlib
import struct
import zlib
from pathlib import Path

import pytest

from ofrak import OFRAKContext
from ofrak.core.dmg import (
    ChunkEntry,
    DmgImage,
    DmgUnpacker,
    KolyBlock,
    MishBlock,
    COMPRESSION_RAW,
    COMPRESSION_ZLIB,
    COMPRESSION_ZERO_FILL,
    COMPRESSION_TERMINATOR,
)
from ofrak.resource import Resource

# Path to test DMG file
TEST_DMG_PATH = Path(__file__).parent / "assets" / "test.dmg"


def create_simple_dmg(data: bytes) -> bytes:
    """
    Create a simple DMG file for testing.

    This creates a minimal valid DMG with:
    - A single blkx entry
    - A mish block with chunks
    - A koly trailer
    """
    # Create chunk entries for the data
    # We'll use raw compression for simplicity
    sector_size = 512
    data_padded = data + b'\x00' * (sector_size - len(data) % sector_size if len(data) % sector_size else 0)
    sector_count = len(data_padded) // sector_size

    # Build mish block
    # Header: signature(4) + version(4) + sector_number(8) + sector_count(8) +
    #         data_offset(8) + buffers_needed(4) + block_descriptors(4) +
    #         reserved(48) + checksum_type(4) + checksum_size(4) + reserved(32)
    chunk_entries = [
        ChunkEntry(
            entry_type=COMPRESSION_RAW,
            sector_number=0,
            sector_count=sector_count,
            compressed_offset=0,
            compressed_length=len(data_padded),
        ),
        ChunkEntry(
            entry_type=COMPRESSION_TERMINATOR,
            sector_number=sector_count,
            sector_count=0,
            compressed_offset=0,
            compressed_length=0,
        ),
    ]

    # Build mish header (204 bytes)
    # Format: signature(4) + version(4) + sector_number(8) + sector_count(8) + data_offset(8)
    #         + buffers_needed(4) + block_descriptors(4) + reserved(48)
    #         + checksum_type(4) + checksum_size(4) + checksum(32) + reserved(76)
    mish_header = struct.pack(
        ">4sIQQQII48xII32x76x",
        b'mish',  # signature
        1,  # version
        0,  # sector_number
        sector_count,  # sector_count
        0,  # data_offset (relative to data fork)
        0,  # buffers_needed
        len(chunk_entries),  # block_descriptors
        0,  # checksum_type
        0,  # checksum_size
    )

    # Build chunk entries (40 bytes each: 4 + 4 padding + 8 + 8 + 8 + 8)
    mish_chunks = b''
    for chunk in chunk_entries:
        mish_chunks += struct.pack(
            ">IxxxxQQQQ",
            chunk.entry_type,
            chunk.sector_number,
            chunk.sector_count,
            chunk.compressed_offset,
            chunk.compressed_length,
        )

    mish_data = mish_header + mish_chunks

    # Create plist with blkx entry
    plist_data = {
        'resource-fork': {
            'blkx': [
                {
                    'Attributes': '0x0050',
                    'CFName': 'whole disk (Apple_HFS : 0)',
                    'Data': base64.b64encode(mish_data),
                    'ID': '0',
                    'Name': 'whole disk (Apple_HFS : 0)',
                }
            ]
        }
    }

    xml_data = plistlib.dumps(plist_data)

    # Build the DMG file structure
    # Layout: [data_fork] [xml_plist] [koly_block]
    data_fork_offset = 0
    data_fork_data = data_padded
    data_fork_length = len(data_fork_data)

    xml_offset = data_fork_offset + data_fork_length
    xml_length = len(xml_data)

    # Build koly block
    # The format string matches the actual koly structure
    koly_data = bytearray(512)

    # Pack the koly block
    # Format: signature(4) + version(4) + header_size(4) + flags(4) +
    #         running_data_fork_offset(8) + data_fork_offset(8) + data_fork_length(8) +
    #         rsrc_fork_offset(8) + rsrc_fork_length(8) + segment_number(4) +
    #         segment_count(4) + segment_id(16) + data_checksum_type(4) +
    #         data_checksum_size(4) + data_checksum(32) + xml_offset(8) +
    #         xml_length(8) + reserved(120) + master_checksum_type(4) +
    #         master_checksum_size(4) + master_checksum(32) + image_variant(4) +
    #         sector_count(8) + reserved(12)

    koly_packed = struct.pack(
        ">4sIIIQQQQQII",
        b'koly',  # signature
        4,  # version
        512,  # header_size
        0,  # flags
        0,  # running_data_fork_offset
        data_fork_offset,  # data_fork_offset
        data_fork_length,  # data_fork_length
        0,  # rsrc_fork_offset
        0,  # rsrc_fork_length
        0,  # segment_number
        1,  # segment_count
    )

    koly_data[0:len(koly_packed)] = koly_packed

    # Add segment_id (16 bytes of zeros at offset 44)
    offset = struct.calcsize(">4sIIIQQQQQII")
    koly_data[offset:offset+16] = b'\x00' * 16
    offset += 16

    # Skip 6 reserved bytes
    offset += 6

    # Add checksum fields
    checksum_data = struct.pack(">II", 0, 0)  # data_checksum_type, data_checksum_size
    koly_data[offset:offset+len(checksum_data)] = checksum_data
    offset += len(checksum_data)

    # Skip 32 bytes for data checksum
    offset += 32

    # Add xml offset and length
    xml_info = struct.pack(">QQ", xml_offset, xml_length)
    koly_data[offset:offset+len(xml_info)] = xml_info
    offset += len(xml_info)

    # Skip 120 reserved bytes
    offset += 120

    # Add master checksum fields and image info
    final_data = struct.pack(
        ">II32sIQIII",
        0,              # master_checksum_type (4 bytes)
        0,              # master_checksum_size (4 bytes)
        b'\x00' * 32,   # master_checksum (32 bytes)
        0,              # image_variant (4 bytes)
        sector_count,   # sector_count (8 bytes)
        0,              # reserved2 (4 bytes)
        0,              # reserved3 (4 bytes)
        0               # reserved4 (4 bytes)
    )
    koly_data[offset:offset+len(final_data)] = final_data

    # Assemble the full DMG
    dmg_file = data_fork_data + xml_data + bytes(koly_data)

    return dmg_file


def create_compressed_dmg(data: bytes) -> bytes:
    """
    Create a DMG file with zlib compression for testing.
    """
    sector_size = 512
    data_padded = data + b'\x00' * (sector_size - len(data) % sector_size if len(data) % sector_size else 0)
    sector_count = len(data_padded) // sector_size

    # Compress the data
    compressed_data = zlib.compress(data_padded)

    # Build chunk entries
    chunk_entries = [
        ChunkEntry(
            entry_type=COMPRESSION_ZLIB,
            sector_number=0,
            sector_count=sector_count,
            compressed_offset=0,
            compressed_length=len(compressed_data),
        ),
        ChunkEntry(
            entry_type=COMPRESSION_TERMINATOR,
            sector_number=sector_count,
            sector_count=0,
            compressed_offset=0,
            compressed_length=0,
        ),
    ]

    # Build mish block (204 bytes header)
    # Format: signature(4) + version(4) + sector_number(8) + sector_count(8) + data_offset(8)
    #         + buffers_needed(4) + block_descriptors(4) + reserved(48)
    #         + checksum_type(4) + checksum_size(4) + checksum(32) + reserved(76)
    mish_header = struct.pack(
        ">4sIQQQII48xII32x76x",
        b'mish',
        1,
        0,
        sector_count,
        0,
        0,
        len(chunk_entries),
        0,
        0,
    )

    mish_chunks = b''
    for chunk in chunk_entries:
        mish_chunks += struct.pack(
            ">IxxxxQQQQ",
            chunk.entry_type,
            chunk.sector_number,
            chunk.sector_count,
            chunk.compressed_offset,
            chunk.compressed_length,
        )

    mish_data = mish_header + mish_chunks

    # Create plist
    plist_data = {
        'resource-fork': {
            'blkx': [
                {
                    'Attributes': '0x0050',
                    'CFName': 'whole disk',
                    'Data': base64.b64encode(mish_data),
                    'ID': '0',
                    'Name': 'whole disk',
                }
            ]
        }
    }

    xml_data = plistlib.dumps(plist_data)

    # Build DMG structure
    data_fork_data = compressed_data
    data_fork_offset = 0
    data_fork_length = len(data_fork_data)

    xml_offset = data_fork_offset + data_fork_length
    xml_length = len(xml_data)

    # Build koly block
    koly_data = bytearray(512)

    koly_packed = struct.pack(
        ">4sIIIQQQQQII",
        b'koly',
        4,
        512,
        0,
        0,
        data_fork_offset,
        data_fork_length,
        0,
        0,
        0,
        1,
    )

    koly_data[0:len(koly_packed)] = koly_packed

    offset = struct.calcsize(">4sIIIQQQQQII")
    koly_data[offset:offset+16] = b'\x00' * 16
    offset += 16 + 6

    checksum_data = struct.pack(">II", 0, 0)
    koly_data[offset:offset+len(checksum_data)] = checksum_data
    offset += len(checksum_data) + 32

    xml_info = struct.pack(">QQ", xml_offset, xml_length)
    koly_data[offset:offset+len(xml_info)] = xml_info
    offset += len(xml_info) + 120

    final_data = struct.pack(
        ">II32sIQIII",
        0,              # master_checksum_type (4 bytes)
        0,              # master_checksum_size (4 bytes)
        b'\x00' * 32,   # master_checksum (32 bytes)
        0,              # image_variant (4 bytes)
        sector_count,   # sector_count (8 bytes)
        0,              # reserved2 (4 bytes)
        0,              # reserved3 (4 bytes)
        0               # reserved4 (4 bytes)
    )
    koly_data[offset:offset+len(final_data)] = final_data

    dmg_file = data_fork_data + xml_data + bytes(koly_data)

    return dmg_file


class TestDmgUnpacker:
    """Tests for DMG unpacker component."""

    async def test_koly_block_identifier(self, ofrak_context: OFRAKContext):
        """Test that KolyBlockIdentifier correctly identifies DMG files."""
        # Create a minimal file with koly signature at the start of the last 512 bytes
        # File structure: [some data] + [512-byte koly block starting with 'koly']
        minimal_dmg = b'\x00' * 1000 + b'koly' + b'\x00' * 508

        root_resource = await ofrak_context.create_root_resource("minimal.dmg", minimal_dmg)
        await root_resource.identify()

        # Should be identified as DmgImage by KolyBlockIdentifier
        assert root_resource.has_tag(DmgImage)

    async def test_koly_block_identifier_rejects_non_dmg(self, ofrak_context: OFRAKContext):
        """Test that KolyBlockIdentifier doesn't misidentify non-DMG files."""
        # Create a file without koly signature
        non_dmg = b'\x00' * 512

        root_resource = await ofrak_context.create_root_resource("not.dmg", non_dmg)
        await root_resource.identify()

        # Should NOT be identified as DmgImage
        assert not root_resource.has_tag(DmgImage)

    async def test_koly_block_identifier_rejects_small_file(self, ofrak_context: OFRAKContext):
        """Test that KolyBlockIdentifier rejects files smaller than 512 bytes."""
        # Create a file smaller than 512 bytes
        small_file = b'koly' + b'\x00' * 100

        root_resource = await ofrak_context.create_root_resource("small.dmg", small_file)
        await root_resource.identify()

        # Should NOT be identified as DmgImage (file too small)
        assert not root_resource.has_tag(DmgImage)

    async def test_unpack_simple_dmg(self, ofrak_context: OFRAKContext):
        """Test unpacking a simple DMG with raw data."""
        test_data = b"Hello, OFRAK! This is a test DMG file."
        dmg_data = create_simple_dmg(test_data)

        # Create resource and identify
        root_resource = await ofrak_context.create_root_resource("test.dmg", dmg_data)
        await root_resource.identify()

        # Should be identified as DmgImage by KolyBlockIdentifier
        assert root_resource.has_tag(DmgImage)

        # Unpack
        await root_resource.unpack()

        # Should have two children: koly block + disk image
        children = list(await root_resource.get_children())
        assert len(children) == 2

        # Find the koly block and disk image
        koly_child = None
        disk_image_child = None
        for child in children:
            if child.has_tag(KolyBlock):
                koly_child = child
            else:
                disk_image_child = child

        assert koly_child is not None, "KolyBlock child should be present"
        assert disk_image_child is not None, "Disk image child should be present"

        # Verify extracted data
        extracted_data = await disk_image_child.get_data()

        # The extracted data should start with our test data
        # (it may be padded to sector boundaries)
        assert extracted_data.startswith(test_data)

    async def test_unpack_compressed_dmg(self, ofrak_context: OFRAKContext):
        """Test unpacking a DMG with zlib compression."""
        test_data = b"This is compressed data in a DMG file. " * 10
        dmg_data = create_compressed_dmg(test_data)

        # Create resource and identify
        root_resource = await ofrak_context.create_root_resource("test_compressed.dmg", dmg_data)
        await root_resource.identify()

        # Should be identified as DmgImage by KolyBlockIdentifier
        assert root_resource.has_tag(DmgImage)

        # Unpack - this will run both ZlibUnpacker and DmgUnpacker
        # since the DMG starts with zlib-compressed data
        await root_resource.unpack()

        # Should have at least one child (may have 2: one from ZlibUnpacker, one from DmgUnpacker)
        children = list(await root_resource.get_children())
        assert len(children) >= 1, "DMG unpacker should create at least one child resource"

        # Find the child created by DmgUnpacker (should contain extracted disk image data)
        dmg_child = None
        for child in children:
            child_data = await child.get_data()
            if len(child_data) > 0 and child_data.startswith(test_data):
                dmg_child = child
                break

        assert dmg_child is not None, "Should find child with extracted DMG data"
        extracted_data = await dmg_child.get_data()

        # The extracted data should start with our test data
        assert extracted_data.startswith(test_data)

    async def test_koly_block_parsing(self):
        """Test parsing of koly block structure."""
        # Create a minimal koly block
        koly_data = bytearray(512)

        # Pack basic koly structure
        koly_packed = struct.pack(
            ">4sIIIQQQQQII",
            b'koly',  # signature
            4,  # version
            512,  # header_size
            0x12345678,  # flags
            0,  # running_data_fork_offset
            0x1000,  # data_fork_offset
            0x2000,  # data_fork_length
            0x3000,  # rsrc_fork_offset
            0x4000,  # rsrc_fork_length
            1,  # segment_number
            2,  # segment_count
        )

        koly_data[0:len(koly_packed)] = koly_packed

        # Add segment_id
        offset = struct.calcsize(">4sIIIQQQQQII")
        koly_data[offset:offset+16] = b'\xAA' * 16
        offset += 16 + 6

        # Add checksum fields
        checksum_data = struct.pack(">II", 1, 32)
        koly_data[offset:offset+len(checksum_data)] = checksum_data
        offset += len(checksum_data) + 32

        # Add XML info
        xml_info = struct.pack(">QQ", 0x5000, 0x1000)
        koly_data[offset:offset+len(xml_info)] = xml_info
        offset += len(xml_info) + 120

        # Add final fields: master_checksum_type, master_checksum_size, master_checksum (32 bytes),
        # image_variant, sector_count, reserved2, reserved3, reserved4
        final_data = struct.pack(
            ">II32sIQIII",
            2,              # master_checksum_type (4 bytes)
            32,             # master_checksum_size (4 bytes)
            b'\x00' * 32,   # master_checksum (32 bytes)
            0,              # image_variant (4 bytes)
            0x10000,        # sector_count (8 bytes)
            0,              # reserved2 (4 bytes)
            0,              # reserved3 (4 bytes)
            0               # reserved4 (4 bytes)
        )
        koly_data[offset:offset+len(final_data)] = final_data

        # Parse the koly block using the static parser method
        koly = DmgUnpacker._parse_koly_block(bytes(koly_data))

        # Verify fields
        assert koly.signature == b'koly'
        assert koly.version == 4
        assert koly.header_size == 512
        assert koly.flags == 0x12345678
        assert koly.data_fork_offset == 0x1000
        assert koly.data_fork_length == 0x2000
        assert koly.rsrc_fork_offset == 0x3000
        assert koly.rsrc_fork_length == 0x4000
        assert koly.segment_number == 1
        assert koly.segment_count == 2
        assert koly.xml_offset == 0x5000
        assert koly.xml_length == 0x1000
        assert koly.sector_count == 0x10000

    async def test_invalid_koly_signature(self):
        """Test that invalid koly signature raises error."""
        invalid_koly = bytearray(512)
        struct.pack_into(">4s", invalid_koly, 0, b'FAKE')

        with pytest.raises(ValueError, match="Invalid koly signature"):
            DmgUnpacker._parse_koly_block(bytes(invalid_koly))

    async def test_invalid_koly_size(self):
        """Test that invalid koly block size raises error."""
        invalid_koly = b'koly' + b'\x00' * 100

        with pytest.raises(ValueError, match="Koly block must be 512 bytes"):
            DmgUnpacker._parse_koly_block(invalid_koly)

    async def test_unpack_simple_compressed_dmg(self, ofrak_context: OFRAKContext):
        """Test unpacking a real simple compressed DMG file without plist metadata.

        This tests the case where xml_length=0 (no plist), and the entire data
        before the koly block is the compressed disk image.
        """
        # Load real DMG file (simple compressed format with xml_length=0)
        with open(TEST_DMG_PATH, 'rb') as f:
            dmg_data = f.read()

        # Create resource and identify
        root_resource = await ofrak_context.create_root_resource("test.dmg", dmg_data)
        await root_resource.identify()

        # Should be identified as DmgImage by KolyBlockIdentifier
        assert root_resource.has_tag(DmgImage)

        # Unpack - may run both DmgUnpacker and ZlibUnpacker
        # since the DMG starts with zlib-compressed data
        await root_resource.unpack()

        # Should have at least one child
        children = list(await root_resource.get_children())
        assert len(children) >= 1, "DMG unpacker should create at least one child resource"

        # Find the child created by DmgUnpacker (the decompressed disk image)
        # It should be the larger of the children (decompressed data is larger)
        children_with_data = []
        for child in children:
            data = await child.get_data()
            children_with_data.append((child, len(data)))

        dmg_child, disk_size = max(children_with_data, key=lambda x: x[1])
        disk_data = await dmg_child.get_data()

        # Simple compressed DMG should have decompressed data
        # The decompressed disk image should be a reasonable size
        assert len(disk_data) > 0, "Disk image data should not be empty"
        # Should be at least 512 bytes (one sector)
        assert len(disk_data) >= 512, "Disk image should be at least one sector"

        # Note: Not all DMG disk images are exact multiples of 512 bytes.
        # The koly block's sector_count field indicates the logical sector count,
        # but the actual decompressed data may include padding or metadata.
