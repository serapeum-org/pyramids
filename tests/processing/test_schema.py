"""Unit tests for :mod:`pyramids.processing.schema`."""

import pytest

from pyramids.processing.schema import ParamSpec, ToolSpec


class TestParamSpec:
    """Tests for the ParamSpec dataclass."""

    def test_init_invalid_param_type_raises(self):
        """An unknown param_type is rejected at construction.

        Test scenario:
            Building a ParamSpec with a type outside PARAM_TYPES raises ValueError
            naming the offending type.
        """
        with pytest.raises(ValueError, match="unknown param_type") as exc:
            ParamSpec("x", "Nonsense")
        assert "Nonsense" in str(exc.value), f"message should name the type: {exc.value}"

    def test_init_choices_on_non_optionlist_raises(self):
        """choices are only valid for an OptionList parameter.

        Test scenario:
            Passing choices for a String param raises ValueError.
        """
        with pytest.raises(ValueError, match="OptionList"):
            ParamSpec("x", "String", choices=("a", "b"))

    @pytest.mark.parametrize(
        "param_type, expected",
        [
            ("Float", True),
            ("Integer", True),
            ("Boolean", True),
            ("String", True),
            ("Field", True),
            ("OptionList", True),
            ("NewFile", True),
            ("Raster", False),
            ("Vector", False),
        ],
    )
    def test_is_serializable_derived(self, param_type, expected):
        """is_serializable is derived from param_type when not overridden.

        Args:
            param_type: The tagged parameter type.
            expected: Whether a value of that type is serializable.

        Test scenario:
            Scalar/file types serialize; in-memory Raster/Vector do not.
        """
        spec = ParamSpec("x", param_type, choices=("a",) if param_type == "OptionList" else None)
        assert spec.is_serializable is expected, f"{param_type} serializable should be {expected}"

    def test_is_serializable_override(self):
        """An explicit serializable flag overrides the derived value.

        Test scenario:
            A Raster param marked serializable=True reports True.
        """
        spec = ParamSpec("x", "Raster", serializable=True)
        assert spec.is_serializable is True, "explicit override should win"

    @pytest.mark.parametrize("value", [0, 3, -2, 2.5, -1.0])
    def test_validate_float_accepts_numbers(self, value):
        """Float accepts ints and floats.

        Args:
            value: A numeric value.

        Test scenario:
            int and float values pass Float validation.
        """
        ParamSpec("x", "Float").validate(value)

    @pytest.mark.parametrize("value", [True, "1.0", [1.0], None, {"a": 1}])
    def test_validate_float_rejects_non_numbers(self, value):
        """Float rejects bools, strings, arrays, None, dicts.

        Args:
            value: A non-numeric value (bool is explicitly excluded).

        Test scenario:
            Passing a numpy-array-like / callable / bool to a Float param raises.
        """
        with pytest.raises(ValueError, match="expects Float"):
            ParamSpec("x", "Float").validate(value)

    def test_validate_integer_rejects_bool_and_float(self):
        """Integer rejects bool and float.

        Test scenario:
            True and 1.5 both fail Integer validation.
        """
        spec = ParamSpec("x", "Integer")
        with pytest.raises(ValueError):
            spec.validate(True)
        with pytest.raises(ValueError):
            spec.validate(1.5)

    def test_validate_boolean(self):
        """Boolean accepts only bool.

        Test scenario:
            True passes; 1 (int) fails.
        """
        spec = ParamSpec("x", "Boolean")
        spec.validate(True)
        with pytest.raises(ValueError):
            spec.validate(1)

    @pytest.mark.parametrize("param_type", ["String", "Field"])
    def test_validate_string_like(self, param_type):
        """String and Field accept str and reject non-str.

        Args:
            param_type: String or Field.

        Test scenario:
            "elevation" passes; 3 fails.
        """
        spec = ParamSpec("x", param_type)
        spec.validate("elevation")
        with pytest.raises(ValueError):
            spec.validate(3)

    def test_validate_optionlist_enforces_choices(self):
        """OptionList enforces membership in choices.

        Test scenario:
            An allowed value passes; a disallowed one raises with the allowed set.
        """
        spec = ParamSpec("x", "OptionList", choices=("degrees", "radians"))
        spec.validate("radians")
        with pytest.raises(ValueError, match=r"one of \['degrees', 'radians'\]"):
            spec.validate("percent")

    def test_validate_newfile_accepts_path(self, tmp_path):
        """NewFile accepts str and os.PathLike.

        Args:
            tmp_path: pytest temp path fixture (an os.PathLike).

        Test scenario:
            Both a string path and a Path object validate.
        """
        spec = ParamSpec("out", "NewFile")
        spec.validate("out.tif")
        spec.validate(tmp_path / "out.tif")

    @pytest.mark.parametrize(
        "param_type, raw, expected",
        [
            ("Float", "2.5", 2.5),
            ("Integer", "3", 3),
            ("Boolean", "true", True),
            ("Boolean", "off", False),
            ("String", "hello", "hello"),
        ],
    )
    def test_coerce_valid(self, param_type, raw, expected):
        """coerce converts a CLI string to the parameter's type.

        Args:
            param_type: The tagged type.
            raw: The raw CLI string.
            expected: The coerced value.

        Test scenario:
            Numeric/boolean/string coercions produce the right Python value.
        """
        result = ParamSpec("x", param_type).coerce(raw)
        assert result == expected, f"coerce({raw!r}) -> {result!r}, expected {expected!r}"

    def test_coerce_boolean_invalid_raises(self):
        """coerce rejects an unparseable boolean string.

        Test scenario:
            "maybe" is not a boolean and raises ValueError.
        """
        with pytest.raises(ValueError, match="not a boolean"):
            ParamSpec("x", "Boolean").coerce("maybe")

    def test_coerce_optionlist_invalid_raises(self):
        """coerce rejects a value outside an OptionList's choices.

        Test scenario:
            "percent" is not among the allowed choices and raises.
        """
        spec = ParamSpec("x", "OptionList", choices=("degrees",))
        with pytest.raises(ValueError, match="not in"):
            spec.coerce("percent")

    def test_help_contains_name_type_and_flag(self):
        """help renders name, type, optionality, default, and description.

        Test scenario:
            A required Field param's help mentions the name, type, and 'required'.
        """
        text = ParamSpec("column", "Field", optional=False, description="Numeric column.").help()
        assert "column" in text and "Field" in text and "required" in text, text


class TestToolSpec:
    """Tests for the ToolSpec dataclass."""

    def test_init_invalid_receiver_raises(self):
        """An invalid receiver type is rejected.

        Test scenario:
            receiver='Blah' raises ValueError listing valid receiver types.
        """
        with pytest.raises(ValueError, match="receiver must be one of"):
            ToolSpec("t", "Blah", "Dataset")

    def test_init_invalid_returns_raises(self):
        """An invalid returns type is rejected.

        Test scenario:
            returns='Blah' raises ValueError.
        """
        with pytest.raises(ValueError, match="returns must be one of"):
            ToolSpec("t", "Dataset", "Blah")

    def test_init_duplicate_param_names_raises(self):
        """Duplicate parameter names are rejected.

        Test scenario:
            Two params both named 'band' raise ValueError.
        """
        dup = (ParamSpec("band", "Integer"), ParamSpec("band", "Integer"))
        with pytest.raises(ValueError, match="duplicate parameter names"):
            ToolSpec("t", "Dataset", "Dataset", dup)

    def test_method_name_defaults_to_name(self):
        """method_name falls back to the tool name.

        Test scenario:
            With no explicit method, method_name equals name; with one, it wins.
        """
        assert ToolSpec("slope", "Dataset", "Dataset").method_name == "slope"
        assert ToolSpec("s", "Dataset", "Dataset", method="do_slope").method_name == "do_slope"

    def test_param_lookup(self):
        """param returns the matching ParamSpec or None.

        Test scenario:
            A known param is found; an unknown one returns None.
        """
        spec = ToolSpec("slope", "Dataset", "Dataset", (ParamSpec("band", "Integer"),))
        assert spec.param("band") is not None, "known param should be found"
        assert spec.param("nope") is None, "unknown param should be None"

    def test_help_lists_params(self):
        """help renders the header and each parameter.

        Test scenario:
            A tool with one param renders its receiver->returns header and the param.
        """
        spec = ToolSpec(
            "slope", "Dataset", "Dataset", (ParamSpec("band", "Integer"),), "Slope."
        )
        text = spec.help()
        assert "Dataset -> Dataset" in text and "band" in text, text

    def test_help_no_params(self):
        """help states '(none)' when a tool has no parameters.

        Test scenario:
            A parameterless tool's help mentions '(none)'.
        """
        assert "(none)" in ToolSpec("t", "Dataset", "Dataset").help()
