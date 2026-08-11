#!/usr/bin/env python3
"""Install Latin Modern Roman fonts under the family name matplotlib/plotnine expects.

The Homebrew cask ``font-latin-modern`` installs the Latin Modern OTF files
under their TeX-internal family name (``LMRoman10``), so matplotlib cannot
resolve ``thesis_theme.FONT_FAMILY`` ("Latin Modern Roman") and silently
falls back to DejaVu Sans with a ``findfont`` warning. This script renames
the ``name`` table of the four weight/style variants to register them as
"Latin Modern Roman" and clears matplotlib's font cache so the change takes
effect immediately.

Usage
-----
    uv run python scripts/install_thesis_fonts.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from fontTools.ttLib import TTFont

FONTS_DIR = Path.home() / "Library" / "Fonts"
FAMILY = "Latin Modern Roman"

# (source file from the font-latin-modern cask) -> (subfamily, preferred subfamily, PostScript name)
VARIANTS = {
    "lmroman10-regular.otf": ("Regular", "Regular", "LatinModernRoman-Regular"),
    "lmroman10-bold.otf": ("Bold", "Bold", "LatinModernRoman-Bold"),
    "lmroman10-italic.otf": ("Italic", "Italic", "LatinModernRoman-Italic"),
    "lmroman10-bolditalic.otf": ("Bold Italic", "Bold Italic", "LatinModernRoman-BoldItalic"),
}


def _ensure_source_fonts() -> None:
    missing = [name for name in VARIANTS if not (FONTS_DIR / name).exists()]
    if not missing:
        return
    if not shutil.which("brew"):
        raise SystemExit(
            f"Missing font files {missing} and Homebrew is not available.\n"
            "Install Latin Modern manually, then re-run this script."
        )
    print("Installing font-latin-modern cask via Homebrew...")
    subprocess.run(["brew", "install", "--cask", "font-latin-modern"], check=True)


def _rename_family(src_name: str, subfamily: str, pref_subfamily: str, ps_name: str) -> Path:
    font = TTFont(FONTS_DIR / src_name)
    name_table = font["name"]

    # (platformID, platEncID, langID) must match the encoding of the records
    # being overwritten -- Macintosh Roman (1,0,0) and Windows Unicode BMP
    # (3,1,0x409) -- otherwise setName() creates new *duplicate* records
    # instead of replacing the originals, and FreeType (used by matplotlib)
    # prefers the untouched (3,1,*) Windows record over the new one.
    for platform_id, enc_id, lang_id in ((1, 0, 0), (3, 1, 0x409)):
        name_table.setName(FAMILY, 1, platform_id, enc_id, lang_id)
        name_table.setName(subfamily, 2, platform_id, enc_id, lang_id)
        name_table.setName(f"{FAMILY} {subfamily}", 4, platform_id, enc_id, lang_id)
        name_table.setName(ps_name, 6, platform_id, enc_id, lang_id)
        name_table.setName(FAMILY, 16, platform_id, enc_id, lang_id)
        name_table.setName(pref_subfamily, 17, platform_id, enc_id, lang_id)
        name_table.setName(f"2.007;GUST;{ps_name}", 3, platform_id, enc_id, lang_id)

    out_path = FONTS_DIR / f"{ps_name}.otf"
    font.save(out_path)
    return out_path


def _refresh_matplotlib_cache() -> None:
    import matplotlib
    import matplotlib.font_manager as fm

    cache_dir = Path(matplotlib.get_cachedir())
    for cache_file in cache_dir.glob("fontlist-*.json"):
        cache_file.unlink()

    # `_load_fontmanager` only returns a fresh instance -- it does not
    # rebind the module-level `fontManager` that `findfont` reads, so we
    # have to reassign it ourselves to pick up the renamed fonts within
    # this process.
    fm.fontManager = fm._load_fontmanager(try_read_cache=False)
    resolved = fm.findfont(fm.FontProperties(family=FAMILY))
    if Path(resolved).stem not in {Path(FONTS_DIR / f"{ps}.otf").stem for *_, ps in VARIANTS.values()}:
        raise SystemExit(f"font resolution failed: matplotlib picked {resolved}")
    print(f"matplotlib resolves '{FAMILY}' -> {resolved}")


def main() -> int:
    _ensure_source_fonts()
    for src_name, (subfamily, pref_subfamily, ps_name) in VARIANTS.items():
        out_path = _rename_family(src_name, subfamily, pref_subfamily, ps_name)
        print(f"wrote {out_path}")
    _refresh_matplotlib_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
