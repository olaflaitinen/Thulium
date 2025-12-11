"""
Comprehensive tests for language profiles.

Validates all 52+ language profiles for consistency, completeness,
and proper configuration. Ensures every language is treated as
a first-class citizen with proper model assignments.
"""

import pytest
from thulium.data.language_profiles import (
    LanguageProfile,
    SUPPORTED_LANGUAGES,
    get_language_profile,
    list_supported_languages,
    get_languages_by_region,
    get_languages_by_script,
    validate_language_profile,
    UnsupportedLanguageError,
)


class TestLanguageProfileBasics:
    """Basic language profile tests."""
    
    def test_minimum_supported_languages(self):
        """Ensure at least 50 languages are supported."""
        languages = list_supported_languages()
        assert len(languages) >= 50, f"Expected 50+ languages, got {len(languages)}"
    
    def test_all_profiles_have_required_fields(self):
        """Validate all profiles have required fields."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            assert profile.code == code, f"Code mismatch for {code}"
            assert profile.name, f"Missing name for {code}"
            assert profile.script, f"Missing script for {code}"
            assert profile.alphabet, f"Empty alphabet for {code}"
            assert profile.direction in ("LTR", "RTL"), f"Invalid direction for {code}"
            assert profile.region, f"Missing region for {code}"
    
    def test_validation_passes_for_all_profiles(self):
        """All profiles should pass validation."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            try:
                validate_language_profile(profile)
            except ValueError as e:
                pytest.fail(f"Validation failed for {code}: {e}")


class TestRegionalCoverage:
    """Tests for regional language coverage."""
    
    def test_caucasus_languages(self):
        """Verify Caucasus region languages."""
        caucasus = get_languages_by_region("Caucasus")
        expected = {"az", "tr", "ka", "hy"}
        assert expected.issubset(set(caucasus)), f"Missing Caucasus languages: {expected - set(caucasus)}"
    
    def test_scandinavian_languages(self):
        """Verify Scandinavian region languages."""
        scandinavia = get_languages_by_region("Scandinavia")
        expected = {"nb", "nn", "sv", "da", "is", "fo", "fi"}
        assert expected.issubset(set(scandinavia)), f"Missing Scandinavian languages: {expected - set(scandinavia)}"
    
    def test_baltic_languages(self):
        """Verify Baltic region languages."""
        baltic = get_languages_by_region("Baltic")
        expected = {"lt", "lv", "et"}
        assert expected.issubset(set(baltic)), f"Missing Baltic languages: {expected - set(baltic)}"
    
    def test_western_europe_languages(self):
        """Verify Western European languages."""
        western = get_languages_by_region("Western Europe")
        expected = {"en", "de", "fr", "es", "pt", "it", "nl"}
        assert expected.issubset(set(western)), f"Missing Western European languages"
    
    def test_eastern_europe_languages(self):
        """Verify Eastern European languages."""
        eastern = get_languages_by_region("Eastern Europe")
        expected = {"pl", "cs", "sk", "hu", "ro", "bg", "ru", "uk"}
        assert expected.issubset(set(eastern)), f"Missing Eastern European languages"
    
    def test_middle_east_languages(self):
        """Verify Middle Eastern languages."""
        middle_east = get_languages_by_region("Middle East")
        expected = {"ar", "fa", "he"}
        assert expected.issubset(set(middle_east)), f"Missing Middle Eastern languages"
    
    def test_south_asia_languages(self):
        """Verify South Asian languages."""
        south_asia = get_languages_by_region("South Asia")
        expected = {"hi", "bn", "ta", "te", "ur"}
        assert expected.issubset(set(south_asia)), f"Missing South Asian languages"
    
    def test_east_asia_languages(self):
        """Verify East Asian languages."""
        east_asia = get_languages_by_region("East Asia")
        expected = {"zh", "ja", "ko"}
        assert expected.issubset(set(east_asia)), f"Missing East Asian languages"


class TestScriptCoverage:
    """Tests for script coverage."""
    
    def test_latin_script_languages(self):
        """Verify Latin script coverage."""
        latin = get_languages_by_script("Latin")
        assert len(latin) >= 30, f"Expected 30+ Latin languages, got {len(latin)}"
    
    def test_cyrillic_script_languages(self):
        """Verify Cyrillic script coverage."""
        cyrillic = get_languages_by_script("Cyrillic")
        expected = {"ru", "uk", "bg", "sr"}
        assert expected.issubset(set(cyrillic)), f"Missing Cyrillic languages"
    
    def test_arabic_script_languages(self):
        """Verify Arabic script coverage."""
        arabic = get_languages_by_script("Arabic")
        expected = {"ar", "fa", "ur"}
        assert expected.issubset(set(arabic)), f"Missing Arabic script languages"
    
    def test_rtl_languages_properly_marked(self):
        """Verify RTL languages have correct direction."""
        rtl_scripts = {"Arabic", "Hebrew"}
        for code, profile in SUPPORTED_LANGUAGES.items():
            if profile.script in rtl_scripts:
                assert profile.direction == "RTL", f"{code} should be RTL"


class TestSpecificLanguages:
    """Tests for specific important languages."""
    
    def test_azerbaijani_profile(self):
        """Verify Azerbaijani language profile."""
        az = get_language_profile("az")
        assert az.name == "Azerbaijani"
        assert az.script == "Latin"
        assert az.region == "Caucasus"
        # Check for schwa character
        assert any(c in az.alphabet for c in "Əə"), "Missing Azerbaijani schwa"
    
    def test_norwegian_variants(self):
        """Verify both Norwegian variants exist."""
        nb = get_language_profile("nb")
        nn = get_language_profile("nn")
        assert nb.name == "Norwegian Bokmal"
        assert nn.name == "Norwegian Nynorsk"
        assert nb.region == "Scandinavia"
        assert nn.region == "Scandinavia"
    
    def test_georgian_profile(self):
        """Verify Georgian profile with unique script."""
        ka = get_language_profile("ka")
        assert ka.name == "Georgian"
        assert ka.script == "Georgian"
        assert ka.direction == "LTR"
    
    def test_armenian_profile(self):
        """Verify Armenian profile."""
        hy = get_language_profile("hy")
        assert hy.name == "Armenian"
        assert hy.script == "Armenian"
    
    def test_chinese_profile(self):
        """Verify Chinese profile."""
        zh = get_language_profile("zh")
        assert "Chinese" in zh.name
        assert zh.script == "Han"
    
    def test_arabic_rtl(self):
        """Verify Arabic is RTL."""
        ar = get_language_profile("ar")
        assert ar.direction == "RTL"
        assert ar.script == "Arabic"


class TestLanguageProfileFunctions:
    """Tests for helper functions."""
    
    def test_get_language_profile_valid(self):
        """Test getting a valid profile."""
        profile = get_language_profile("en")
        assert profile.code == "en"
        assert profile.name == "English"
    
    def test_get_language_profile_invalid(self):
        """Test getting an invalid profile raises error."""
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            get_language_profile("xyz")
        assert "xyz" in str(exc_info.value)
    
    def test_list_supported_languages_sorted(self):
        """Verify language list is sorted."""
        languages = list_supported_languages()
        assert languages == sorted(languages)
    
    def test_vocab_size_positive(self):
        """All profiles should have positive vocab size."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            assert profile.get_vocab_size() > 0, f"Zero vocab for {code}"
    
    def test_char_mapping_consistency(self):
        """Character mappings should be consistent."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            char_to_idx = profile.get_char_to_idx()
            idx_to_char = profile.get_idx_to_char()
            
            for char, idx in char_to_idx.items():
                assert idx_to_char[idx] == char, f"Mapping mismatch for {code}"


class TestModelProfileAssignments:
    """Tests for model profile assignments."""
    
    def test_all_profiles_have_model_profile(self):
        """All profiles should have a model_profile field."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            assert hasattr(profile, 'model_profile'), f"Missing model_profile for {code}"
            assert profile.model_profile, f"Empty model_profile for {code}"
    
    def test_latin_languages_use_latin_model(self):
        """Latin script languages should use Latin multilingual model."""
        for code, profile in SUPPORTED_LANGUAGES.items():
            if profile.script == "Latin":
                assert "latin" in profile.model_profile.lower(), \
                    f"{code} should use Latin model, got {profile.model_profile}"
