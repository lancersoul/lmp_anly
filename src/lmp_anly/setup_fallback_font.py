from pathlib import Path

from matplotlib.font_manager import fontManager


def setup_fallback_fonts():
    asset_dir = Path(__file__).parent / "assets"
    chinese_font_path = asset_dir / "FandolSong-Regular.otf"
    english_regular_font_path = asset_dir / "texgyretermes-regular.otf"
    english_italic_font_path = asset_dir / "texgyretermes-italic.otf"
    english_bold_font_path = asset_dir / "texgyretermes-bold.otf"
    english_bolditalic_font_path = asset_dir / "texgyretermes-bolditalic.otf"
    try:
        fontManager.addfont(str(chinese_font_path))
        fontManager.addfont(str(english_regular_font_path))
        fontManager.addfont(str(english_italic_font_path))
        fontManager.addfont(str(english_bold_font_path))
        fontManager.addfont(str(english_bolditalic_font_path))
    except OSError:
        print("builtin font not found")

