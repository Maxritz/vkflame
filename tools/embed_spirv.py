#!/usr/bin/env python3
"""
Usage: python tools/embed_spirv.py <spirv_dir> <output_cpp>

Scans spirv_dir for *.spv files and embeds them as C++ byte arrays
with a lookup table for runtime access.
"""
import sys
import os
from pathlib import Path


def embed_spirv(spirv_dir: str, output_cpp: str) -> None:
    spirv_path = Path(spirv_dir)
    spv_files = sorted(spirv_path.glob("*.spv"))

    if not spv_files:
        print(f"[embed_spirv] WARNING: no .spv files found in {spirv_dir}", file=sys.stderr)

    lines = []
    lines.append("// auto-generated — do not edit")
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("")

    kernel_names = []
    for spv_file in spv_files:
        kernel_name = spv_file.stem  # filename without .spv
        data = spv_file.read_bytes()
        kernel_names.append(kernel_name)

        hex_bytes = [f"0x{b:02x}" for b in data]
        # 16 bytes per line
        chunks = [hex_bytes[i:i+16] for i in range(0, len(hex_bytes), 16)]
        hex_lines = ",\n    ".join(", ".join(chunk) for chunk in chunks)

        lines.append(f"extern \"C\" const uint8_t  vkf_spv_{kernel_name}[] = {{")
        lines.append(f"    {hex_lines}")
        lines.append("};")
        lines.append(f"extern \"C\" const uint32_t vkf_spv_{kernel_name}_len = {len(data)};")
        lines.append("")

    # Lookup table
    lines.append("struct VKFSpvEntry { const char* name; const uint8_t* data; uint32_t len; };")
    lines.append("extern \"C\" const VKFSpvEntry vkf_spirv_table[] = {")
    for name in kernel_names:
        lines.append(f'    {{ "{name}", vkf_spv_{name}, vkf_spv_{name}_len }},')
    lines.append("    { nullptr, nullptr, 0 }")
    lines.append("};")
    lines.append("")

    output = "\n".join(lines)
    Path(output_cpp).write_text(output, encoding="utf-8")
    print(f"[embed_spirv] wrote {len(kernel_names)} kernels to {output_cpp}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python embed_spirv.py <spirv_dir> <output_cpp>")
        sys.exit(1)
    embed_spirv(sys.argv[1], sys.argv[2])
