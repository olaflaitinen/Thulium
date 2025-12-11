#!/usr/bin/env python3
"""
Caucasus Region Languages Recognition Example

This example demonstrates Thulium's support for Caucasus region languages:
- Azerbaijani (Latin script with extended characters)
- Turkish (Latin script)
- Georgian (unique Mkhedruli script)
- Armenian (unique Armenian script)

Each language is treated as a first-class citizen with dedicated
or appropriately shared model configurations.
"""

from pathlib import Path


def main():
    """Demonstrate Caucasus region language recognition."""
    
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
    print("Thulium - Caucasus Region Languages Recognition")
    print("=" * 60)
    print()
    
    # List Caucasus languages
    caucasus = get_languages_by_region("Caucasus")
    print(f"Supported Caucasus languages ({len(caucasus)}):")
    for code in caucasus:
        profile = get_language_profile(code)
        print(f"  [{code}] {profile.name}")
        print(f"       Script: {profile.script}")
        print(f"       Model: {profile.model_profile}")
    print()
    
    # Azerbaijani-specific demonstration
    print("Azerbaijani Language Details:")
    print("-" * 60)
    az_profile = get_language_profile("az")
    
    # Show special characters
    special_chars = [c for c in az_profile.alphabet if c in "ƏəĞğIıÖöŞşÜüÇç"]
    print(f"Special characters: {' '.join(special_chars)}")
    print(f"Total alphabet size: {len(az_profile.alphabet)} characters")
    print(f"Notes: {az_profile.notes}")
    print()
    
    # Georgian demonstration
    print("Georgian Language Details:")
    print("-" * 60)
    ka_profile = get_language_profile("ka")
    print(f"Script: {ka_profile.script} (Mkhedruli)")
    print(f"Alphabet size: {len(ka_profile.alphabet)} characters")
    print(f"Model: {ka_profile.model_profile} (specialized)")
    print()
    
    # Armenian demonstration
    print("Armenian Language Details:")
    print("-" * 60)
    hy_profile = get_language_profile("hy")
    print(f"Script: {hy_profile.script}")
    print(f"Alphabet size: {len(hy_profile.alphabet)} characters")
    print(f"Model: {hy_profile.model_profile} (specialized)")
    print()
    
    # Example usage
    print("Usage Examples:")
    print("-" * 60)
    print("""
    # Azerbaijani recognition
    result = recognize_image("az_sample.png", language="az")
    
    # Georgian recognition
    result = recognize_image("ka_sample.png", language="ka")
    
    # Armenian recognition
    result = recognize_image("hy_sample.png", language="hy")
    
    # CLI usage
    thulium recognize sample.png --language az
    thulium recognize sample.png --language ka --pipeline htr_georgian
    """)


if __name__ == "__main__":
    main()
