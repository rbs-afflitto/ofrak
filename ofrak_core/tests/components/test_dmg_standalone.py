#!/usr/bin/env python3
"""
Standalone test for DMG unpacker that doesn't require OFRAK license.
Tests basic parsing and decompression functionality.
"""

import base64
import plistlib
import struct
import zlib

from ofrak.core.dmg import (
    ChunkEntry,
    KolyBlock,
    MishBlock,
    COMPRESSION_RAW,
    COMPRESSION_ZLIB,
    COMPRESSION_ZERO_FILL,
    COMPRESSION_TERMINATOR,
)


def test_koly_block_parsing():
    """Test parsing of koly block structure."""
    print("Testing koly block parsing...")

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
    offset += 16

    # Add checksum fields (128 bytes for data_checksum)
    checksum_data = struct.pack(">II", 1, 128)
    koly_data[offset:offset+len(checksum_data)] = checksum_data
    offset += len(checksum_data) + 128

    # Add XML info
    xml_info = struct.pack(">QQ", 0x5000, 0x1000)
    koly_data[offset:offset+len(xml_info)] = xml_info
    offset += len(xml_info) + 120

    # Add final fields: master_checksum_type, master_checksum_size, (128 bytes checksum), image_variant, sector_count, reserved fields
    final_data = struct.pack(">II", 2, 128)  # master_checksum_type, master_checksum_size
    koly_data[offset:offset+len(final_data)] = final_data
    offset += len(final_data)

    # Skip 128 bytes for master_checksum
    offset += 128

    # Add image_variant, sector_count, and reserved fields
    image_and_sector = struct.pack(">IQIII", 0, 0x10000, 0, 0, 0)  # image_variant, sector_count, 3 reserved fields
    koly_data[offset:offset+len(image_and_sector)] = image_and_sector

    # Parse the koly block
    koly = KolyBlock.from_bytes(bytes(koly_data))

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

    print("✓ Koly block parsing successful")


def test_invalid_koly_signature():
    """Test that invalid koly signature raises error."""
    print("Testing invalid koly signature...")

    invalid_koly = bytearray(512)
    struct.pack_into(">4s", invalid_koly, 0, b'FAKE')

    try:
        KolyBlock.from_bytes(bytes(invalid_koly))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid koly signature" in str(e)

    print("✓ Invalid signature correctly rejected")


def test_invalid_koly_size():
    """Test that invalid koly block size raises error."""
    print("Testing invalid koly size...")

    invalid_koly = b'koly' + b'\x00' * 100

    try:
        KolyBlock.from_bytes(invalid_koly)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Koly block must be 512 bytes" in str(e)

    print("✓ Invalid size correctly rejected")


def test_mish_block_parsing():
    """Test parsing of mish block structure."""
    print("Testing mish block parsing...")

    # Build mish header (204 bytes total)
    # We build it manually to ensure correct size
    mish_header = bytearray(204)
    offset = 0

    # Signature, version
    struct.pack_into(">4sI", mish_header, offset, b'mish', 1)
    offset += 8

    # sector_number, sector_count, data_offset
    struct.pack_into(">QQQ", mish_header, offset, 0, 100, 0)
    offset += 24

    # buffers_needed, block_descriptors
    struct.pack_into(">II", mish_header, offset, 0, 2)
    offset += 8

    # 48 bytes reserved
    offset += 48

    # checksum_type, checksum_size
    struct.pack_into(">II", mish_header, offset, 0, 0)
    offset += 8

    # 32 bytes checksum
    offset += 32

    # Remaining bytes to fill to 204 (should be 204 - 120 = 84 bytes)
    # offset should be 120 at this point, so we need 84 more bytes

    mish_header = bytes(mish_header)

    # Build chunk entries (40 bytes each: 4 + 4 padding + 8 + 8 + 8 + 8)
    chunk1 = struct.pack(
        ">IxxxxQQQQ",
        COMPRESSION_RAW,  # entry_type
        0,  # sector_number
        50,  # sector_count
        0,  # compressed_offset
        25600,  # compressed_length
    )

    chunk2 = struct.pack(
        ">IxxxxQQQQ",
        COMPRESSION_TERMINATOR,  # entry_type
        50,  # sector_number
        0,  # sector_count
        0,  # compressed_offset
        0,  # compressed_length
    )

    mish_data = mish_header + chunk1 + chunk2

    # Parse the mish block
    mish = MishBlock.from_bytes(mish_data)

    # Verify fields
    assert mish.signature == b'mish'
    assert mish.version == 1
    assert mish.sector_number == 0
    assert mish.sector_count == 100
    assert mish.data_offset == 0
    assert mish.block_descriptors == 2
    assert len(mish.chunks) == 2

    # Verify first chunk
    assert mish.chunks[0].entry_type == COMPRESSION_RAW
    assert mish.chunks[0].sector_number == 0
    assert mish.chunks[0].sector_count == 50
    assert mish.chunks[0].compressed_length == 25600

    # Verify second chunk
    assert mish.chunks[1].entry_type == COMPRESSION_TERMINATOR

    print("✓ Mish block parsing successful")


def test_chunk_decompression():
    """Test chunk decompression for different compression types."""
    print("Testing chunk decompression...")

    # Test data
    test_data = b"Hello, OFRAK! This is test data for DMG unpacker."

    # Test zero-fill
    chunk_zero = ChunkEntry(
        entry_type=COMPRESSION_ZERO_FILL,
        sector_number=0,
        sector_count=10,
        compressed_offset=0,
        compressed_length=0,
    )

    # Zero fill should return zeros
    from ofrak.core.dmg import DmgUnpacker
    result = DmgUnpacker._decompress_chunk(b'', chunk_zero, 0)
    assert result == b'\x00' * (10 * 512)

    print("✓ Zero-fill decompression successful")

    # Test raw data
    chunk_raw = ChunkEntry(
        entry_type=COMPRESSION_RAW,
        sector_number=0,
        sector_count=1,
        compressed_offset=0,
        compressed_length=len(test_data),
    )

    result = DmgUnpacker._decompress_chunk(test_data, chunk_raw, 0)
    assert result == test_data

    print("✓ Raw decompression successful")

    # Test zlib compression
    compressed_data = zlib.compress(test_data)
    chunk_zlib = ChunkEntry(
        entry_type=COMPRESSION_ZLIB,
        sector_number=0,
        sector_count=1,
        compressed_offset=0,
        compressed_length=len(compressed_data),
    )

    result = DmgUnpacker._decompress_chunk(compressed_data, chunk_zlib, 0)
    assert result == test_data

    print("✓ zlib decompression successful")


def main():
    """Run all standalone tests."""
    print("=" * 60)
    print("DMG Unpacker Standalone Tests")
    print("=" * 60)
    print()

    try:
        test_koly_block_parsing()
        test_invalid_koly_signature()
        test_invalid_koly_size()
        test_mish_block_parsing()
        test_chunk_decompression()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
