#!/usr/bin/env python3
"""
Scandinavian Languages Recognition Example

This example demonstrates Thulium's support for Scandinavian languages:
- Norwegian (Bokmal and Nynorsk)
- Swedish
- Danish
- Icelandic
- Faroese
- Finnish

All Scandinavian languages use the Latin script with extended characters
(ae, o-slash, a-ring, eth, thorn, etc.) and are supported by the shared
Latin multilingual model.
"""

from pathlib import Path


def main():
    """Demonstrate Scandinavian language recognition."""
    
    # Import Thulium API
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
    print("Thulium - Scandinavian Languages Recognition")
    print("=" * 60)
    print()
    
    # List Scandinavian languages
    scandinavian = get_languages_by_region("Scandinavia")
    print(f"Supported Scandinavian languages ({len(scandinavian)}):")
    for code in scandinavian:
        profile = get_language_profile(code)
        print(f"  [{code}] {profile.name}")
        print(f"       Script: {profile.script}")
        print(f"       Alphabet size: {len(profile.alphabet)} characters")
    print()
    
    # Example recognition for each language
    example_images = {
        "nb": "examples/data/norwegian_bokmal_sample.png",
        "nn": "examples/data/norwegian_nynorsk_sample.png",
        "sv": "examples/data/swedish_sample.png",
        "da": "examples/data/danish_sample.png",
        "is": "examples/data/icelandic_sample.png",
        "fo": "examples/data/faroese_sample.png",
        "fi": "examples/data/finnish_sample.png",
    }
    
    print("Recognition Examples:")
    print("-" * 60)
    
    for code, image_path in example_images.items():
        profile = get_language_profile(code)
        
        if Path(image_path).exists():
            result = recognize_image(image_path, language=code)
            print(f"\n[{code}] {profile.name}:")
            print(f"  Text: {result.full_text}")
            print(f"  Confidence: {result.confidence:.2f}")
        else:
            print(f"\n[{code}] {profile.name}: (sample image not found)")
            print(f"  To test, provide: {image_path}")
    
    print()
    print("=" * 60)
    print("For custom images, use:")
    print('  result = recognize_image("your_image.png", language="sv")')
    print("=" * 60)


if __name__ == "__main__":
    main()
