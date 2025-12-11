#!/usr/bin/env python3
"""
Baltic Languages Recognition Example

This example demonstrates Thulium's support for Baltic languages:
- Lithuanian
- Latvian
- Estonian (Finno-Ugric, but geographically Baltic)

All Baltic languages use the Latin script with extended diacritical
marks (ogonek, macron, caron, cedilla, tilde) and are supported
by the shared Latin multilingual model.
"""

from pathlib import Path


def main():
    """Demonstrate Baltic language recognition."""
    
    try:
        from thulium.api import recognize_image
        from thulium.data.language_profiles import (
            get_language_profile,
            get_languages_by_region,
        )
    except ImportError:
        print("Please install Thulium: pip install thulium")
        return
    
    print("=" * 60)
    print("Thulium - Baltic Languages Recognition")
    print("=" * 60)
    print()
    
    # List Baltic languages
    baltic = get_languages_by_region("Baltic")
    print(f"Supported Baltic languages ({len(baltic)}):")
    
    for code in baltic:
        profile = get_language_profile(code)
        print(f"\n  [{code}] {profile.name}")
        print(f"       Script: {profile.script}")
        
        # Show extended characters
        special = [c for c in profile.alphabet 
                   if c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:;\"'()[] "]
        if special:
            print(f"       Extended chars: {' '.join(sorted(set(special))[:15])}")
        
        if profile.notes:
            print(f"       Notes: {profile.notes}")
    
    print()
    print("=" * 60)
    print("Baltic Language Characteristics:")
    print("-" * 60)
    print("""
    Lithuanian (lt):
      - Uses ogonek, caron, and macron diacritics
      - Example letters: a-ogonek, c-caron, e-ogonek, s-caron, z-caron
      - One of the most archaic Indo-European languages
    
    Latvian (lv):
      - Uses macron, cedilla, and caron diacritics
      - Example letters: a-macron, c-cedilla, g-cedilla, k-cedilla
      - Related to Lithuanian
    
    Estonian (et):
      - Uses umlaut and tilde diacritics
      - Example letters: a-umlaut, o-tilde, o-umlaut, u-umlaut
      - Finno-Ugric (not Indo-European)
    """)
    
    print("Usage:")
    print("-" * 60)
    print("""
    from thulium.api import recognize_image
    
    # Lithuanian
    result = recognize_image("lt_sample.png", language="lt")
    
    # Latvian
    result = recognize_image("lv_sample.png", language="lv")
    
    # Estonian
    result = recognize_image("et_sample.png", language="et")
    """)


if __name__ == "__main__":
    main()
