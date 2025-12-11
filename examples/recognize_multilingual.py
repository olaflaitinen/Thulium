#!/usr/bin/env python3
"""
Multilingual Recognition Example

This example demonstrates Thulium's comprehensive multilingual
capabilities across all 52+ supported languages, organized by
region and script family.
"""


def main():
    """Demonstrate comprehensive multilingual support."""
    
    try:
        from thulium.data.language_profiles import (
            SUPPORTED_LANGUAGES,
            get_language_profile,
            get_languages_by_region,
            get_languages_by_script,
            list_supported_languages,
        )
    except ImportError:
        print("Please install Thulium: pip install thulium")
        return
    
    print("=" * 70)
    print("Thulium Multilingual Handwriting Recognition")
    print("=" * 70)
    print()
    
    # Count totals
    all_languages = list_supported_languages()
    print(f"Total Supported Languages: {len(all_languages)}")
    print()
    
    # Group by region
    regions = {}
    for code, profile in SUPPORTED_LANGUAGES.items():
        region = profile.region
        if region not in regions:
            regions[region] = []
        regions[region].append(code)
    
    print("Languages by Region:")
    print("-" * 70)
    for region, codes in sorted(regions.items()):
        print(f"\n{region} ({len(codes)} languages):")
        for code in sorted(codes):
            profile = get_language_profile(code)
            print(f"  [{code:8s}] {profile.name:25s} ({profile.script})")
    
    print()
    print("=" * 70)
    print("Languages by Script:")
    print("-" * 70)
    
    scripts = {}
    for code, profile in SUPPORTED_LANGUAGES.items():
        script = profile.script
        if script not in scripts:
            scripts[script] = []
        scripts[script].append(code)
    
    for script, codes in sorted(scripts.items(), key=lambda x: -len(x[1])):
        direction = "RTL" if script in ("Arabic", "Hebrew") else "LTR"
        print(f"\n{script} ({len(codes)} languages, {direction}):")
        print(f"  {', '.join(sorted(codes))}")
    
    print()
    print("=" * 70)
    print("Model Profiles:")
    print("-" * 70)
    
    model_profiles = {}
    for code, profile in SUPPORTED_LANGUAGES.items():
        mp = profile.model_profile
        if mp not in model_profiles:
            model_profiles[mp] = []
        model_profiles[mp].append(code)
    
    for profile_name, codes in sorted(model_profiles.items()):
        print(f"\n{profile_name}:")
        print(f"  Languages: {', '.join(sorted(codes))}")
    
    print()
    print("=" * 70)
    print("Quick Start Examples:")
    print("-" * 70)
    print("""
    from thulium.api import recognize_image
    
    # European Languages
    result_en = recognize_image("english.png", language="en")
    result_de = recognize_image("german.png", language="de")
    result_fr = recognize_image("french.png", language="fr")
    
    # Scandinavian
    result_sv = recognize_image("swedish.png", language="sv")
    result_nb = recognize_image("norwegian.png", language="nb")
    
    # Baltic
    result_lt = recognize_image("lithuanian.png", language="lt")
    
    # Caucasus
    result_az = recognize_image("azerbaijani.png", language="az")
    result_ka = recognize_image("georgian.png", language="ka")
    
    # Middle East (RTL)
    result_ar = recognize_image("arabic.png", language="ar")
    result_he = recognize_image("hebrew.png", language="he")
    
    # South Asia
    result_hi = recognize_image("hindi.png", language="hi")
    result_bn = recognize_image("bengali.png", language="bn")
    
    # East Asia
    result_zh = recognize_image("chinese.png", language="zh")
    result_ja = recognize_image("japanese.png", language="ja")
    result_ko = recognize_image("korean.png", language="ko")
    """)


if __name__ == "__main__":
    main()
