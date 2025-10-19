# tests/unit/visualization/test_06_diagram_generators.py
"""
Unit tests for diagram_generators module.

Tests specialized diagram generation functions with code compliance.
"""

import numpy as np
import pytest


# ============================================================================
# DESIGN CODE ENUM TESTS
# ============================================================================


@pytest.mark.unit
class TestDesignCode:
    """Test DesignCode enum."""

    def test_all_codes_present(self, ensure_imports):
        """DesignCode should have all major codes."""
        from pymemorial.visualization import DesignCode

        required_codes = ["NBR8800", "NBR6118", "EN1993", "AISC360", "ACI318"]
        
        for code in required_codes:
            assert hasattr(DesignCode, code)

    def test_code_values(self, ensure_imports):
        """DesignCode values should be descriptive."""
        from pymemorial.visualization import DesignCode

        assert "8800" in DesignCode.NBR8800.value
        assert "EN 1993" in DesignCode.EN1993.value
        assert "AISC" in DesignCode.AISC360.value


# ============================================================================
# DATACLASS TESTS
# ============================================================================


@pytest.mark.unit
class TestPMDiagramParams:
    """Test PMDiagramParams dataclass."""

    def test_default_values(self, ensure_imports):
        """PMDiagramParams should have sensible defaults."""
        from pymemorial.visualization import PMDiagramParams

        params = PMDiagramParams()
        assert params.n_points == 50
        assert params.include_tension is True
        assert params.normalize is True
        assert params.safety_factor == 1.0

    def test_custom_values(self, ensure_imports):
        """PMDiagramParams should accept custom values."""
        from pymemorial.visualization import PMDiagramParams, DesignCode

        params = PMDiagramParams(
            n_points=100,
            include_tension=False,
            code=DesignCode.EN1993,
            safety_factor=1.5,
        )
        
        assert params.n_points == 100
        assert params.include_tension is False
        assert params.code == DesignCode.EN1993
        assert params.safety_factor == 1.5

    def test_immutable(self, ensure_imports):
        """PMDiagramParams should be immutable."""
        from pymemorial.visualization import PMDiagramParams

        params = PMDiagramParams(n_points=50)
        
        with pytest.raises(AttributeError):
            params.n_points = 100


@pytest.mark.unit
class TestMomentCurvatureParams:
    """Test MomentCurvatureParams dataclass."""

    def test_default_values(self, ensure_imports):
        """MomentCurvatureParams should have sensible defaults."""
        from pymemorial.visualization import MomentCurvatureParams

        params = MomentCurvatureParams()
        assert params.target_curvature == 0.01
        assert params.n_steps == 100
        assert params.mark_yield is True
        assert params.calculate_ductility is True

    def test_custom_values(self, ensure_imports):
        """MomentCurvatureParams should accept custom values."""
        from pymemorial.visualization import MomentCurvatureParams

        params = MomentCurvatureParams(
            target_curvature=0.02,
            n_steps=200,
            include_unloading=True,
        )
        
        assert params.target_curvature == 0.02
        assert params.n_steps == 200
        assert params.include_unloading is True


# ============================================================================
# P-M ENVELOPE GENERATION TESTS
# ============================================================================


@pytest.mark.unit
class TestPMEnvelopeGeneration:
    """Test P-M interaction envelope generation."""

    def test_generate_pm_rectangular(self, ensure_imports):
        """Generate P-M envelope for rectangular section."""
        from pymemorial.visualization import generate_pm_interaction_envelope

        p, m = generate_pm_interaction_envelope(
            p_nominal=1000,  # kN
            m_nominal=250,   # kN·m
            section_type="rectangular",
        )
        
        assert len(p) > 10
        assert len(m) > 10
        assert len(p) == len(m)
        
        # Normalized values should be in [0, 1] range (with small tension)
        assert np.all(m >= 0)
        assert np.all(p >= -0.5)  # Allow small tension
        assert np.all(p <= 1.1)   # Allow small excess

    def test_generate_pm_i_section(self, ensure_imports):
        """Generate P-M envelope for I-section."""
        from pymemorial.visualization import generate_pm_interaction_envelope

        p, m = generate_pm_interaction_envelope(
            p_nominal=2000,
            m_nominal=500,
            section_type="i_section",
        )
        
        assert len(p) == len(m)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(m))

    def test_generate_pm_circular(self, ensure_imports):
        """Generate P-M envelope for circular section."""
        from pymemorial.visualization import generate_pm_interaction_envelope

        p, m = generate_pm_interaction_envelope(
            p_nominal=1500,
            m_nominal=300,
            section_type="circular",
        )
        
        assert len(p) == len(m)

    def test_generate_pm_invalid_section(self, ensure_imports):
        """Invalid section type should raise error."""
        from pymemorial.visualization import generate_pm_interaction_envelope

        with pytest.raises(ValueError, match="Unknown section_type"):
            generate_pm_interaction_envelope(
                p_nominal=1000,
                m_nominal=250,
                section_type="invalid_type",
            )

    def test_generate_pm_with_params(self, ensure_imports):
        """Generate P-M with custom parameters."""
        from pymemorial.visualization import (
            generate_pm_interaction_envelope,
            PMDiagramParams,
        )

        params = PMDiagramParams(
            n_points=30,
            include_tension=False,
            safety_factor=1.5,
        )
        
        p, m = generate_pm_interaction_envelope(
            p_nominal=1000,
            m_nominal=250,
            params=params,
        )
        
        # Should be affected by safety factor
        assert np.max(p) < 1.0  # Reduced by factor
        assert len(p) >= 30

    def test_generate_pm_denormalized(self, ensure_imports):
        """Generate P-M with actual values (not normalized)."""
        from pymemorial.visualization import (
            generate_pm_interaction_envelope,
            PMDiagramParams,
        )

        params = PMDiagramParams(normalize=False)
        
        p, m = generate_pm_interaction_envelope(
            p_nominal=1000,
            m_nominal=250,
            params=params,
        )
        
        # Should have actual values
        assert np.max(p) > 10  # Much larger than 1
        assert np.max(m) > 10


# ============================================================================
# MOMENT-CURVATURE GENERATION TESTS
# ============================================================================


@pytest.mark.unit
class TestMomentCurvatureGeneration:
    """Test moment-curvature response generation."""

    def test_generate_mk_basic(self, ensure_imports):
        """Generate basic moment-curvature response."""
        from pymemorial.visualization import generate_moment_curvature_response

        kappa, m = generate_moment_curvature_response(
            m_yield=200,       # kN·m
            m_ultimate=250,    # kN·m
            kappa_yield=0.003, # 1/m
            kappa_ultimate=0.015,
        )
        
        assert len(kappa) > 50
        assert len(m) > 50
        assert len(kappa) == len(m)
        
        # Should start at zero
        assert kappa[0] == 0
        assert m[0] == 0
        
        # Should be monotonic increasing
        assert np.all(np.diff(kappa) >= 0)

    def test_generate_mk_with_unloading(self, ensure_imports):
        """Generate M-κ with unloading branch."""
        from pymemorial.visualization import (
            generate_moment_curvature_response,
            MomentCurvatureParams,
        )

        params = MomentCurvatureParams(include_unloading=True)
        
        kappa, m = generate_moment_curvature_response(
            m_yield=200,
            m_ultimate=250,
            kappa_yield=0.003,
            kappa_ultimate=0.015,
            params=params,
        )
        
        # Should have unloading region
        assert len(kappa) > 100
        assert np.max(kappa) > 0.015  # Extends beyond ultimate

    def test_generate_mk_custom_steps(self, ensure_imports):
        """Generate M-κ with custom number of steps."""
        from pymemorial.visualization import (
            generate_moment_curvature_response,
            MomentCurvatureParams,
        )

        params = MomentCurvatureParams(n_steps=50)
        
        kappa, m = generate_moment_curvature_response(
            m_yield=200,
            m_ultimate=250,
            kappa_yield=0.003,
            kappa_ultimate=0.015,
            params=params,
        )
        
        # Should have approximately n_steps points
        assert 48 <= len(kappa) <= 52  # Within ±2 of target



# ============================================================================
# CODE-SPECIFIC DIAGRAM TESTS
# ============================================================================


@pytest.mark.unit
class TestCodeSpecificDiagrams:
    """Test code-specific diagram creation."""

    def test_create_pm_with_nbr8800(self, ensure_imports):
        """Create P-M diagram with NBR 8800."""
        from pymemorial.visualization import (
            create_visualizer,
            create_pm_diagram_with_code,
            DesignCode,
        )

        viz = create_visualizer()
        
        fig = create_pm_diagram_with_code(
            engine=viz,
            p_nominal=1000,
            m_nominal=250,
            code=DesignCode.NBR8800,
        )
        
        assert fig is not None

    def test_create_pm_with_en1993(self, ensure_imports):
        """Create P-M diagram with EN 1993."""
        from pymemorial.visualization import (
            create_visualizer,
            create_pm_diagram_with_code,
            DesignCode,
        )

        viz = create_visualizer()
        
        fig = create_pm_diagram_with_code(
            engine=viz,
            p_nominal=1000,
            m_nominal=250,
            code=DesignCode.EN1993,
        )
        
        assert fig is not None

    def test_create_pm_with_design_point(self, ensure_imports):
        """Create P-M diagram with design point."""
        from pymemorial.visualization import (
            create_visualizer,
            create_pm_diagram_with_code,
            DesignCode,
        )

        viz = create_visualizer()
        
        fig = create_pm_diagram_with_code(
            engine=viz,
            p_nominal=1000,
            m_nominal=250,
            design_point=(150, 600),
            code=DesignCode.NBR8800,
        )
        
        assert fig is not None

    def test_create_pm_with_custom_config(self, ensure_imports):
        """Create P-M diagram with custom configuration."""
        from pymemorial.visualization import (
            create_visualizer,
            create_pm_diagram_with_code,
            PlotConfig,
            DesignCode,
        )

        viz = create_visualizer()
        config = PlotConfig(title="Custom P-M Diagram", width=1000, height=800)
        
        fig = create_pm_diagram_with_code(
            engine=viz,
            p_nominal=1000,
            m_nominal=250,
            config=config,
            code=DesignCode.NBR8800,
        )
        
        assert fig is not None


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


@pytest.mark.unit
class TestDuctilityCalculation:
    """Test ductility calculation utilities."""

    def test_calculate_ductility_high(self, ensure_imports):
        """Calculate high ductility."""
        from pymemorial.visualization import calculate_ductility

        metrics = calculate_ductility(
            kappa_yield=0.003,
            kappa_ultimate=0.015,
        )
        
        assert "mu" in metrics
        assert "classification" in metrics
        assert "adequate" in metrics
        
        # mu = 0.015 / 0.003 = 5.0
        assert abs(metrics["mu"] - 5.0) < 0.01
        assert metrics["classification"] == "high"
        assert metrics["adequate"] is True

    def test_calculate_ductility_medium(self, ensure_imports):
        """Calculate medium ductility."""
        from pymemorial.visualization import calculate_ductility

        metrics = calculate_ductility(
            kappa_yield=0.003,
            kappa_ultimate=0.0105,  # mu = 3.5 (clearly medium and adequate)
        )

        assert abs(metrics["mu"] - 3.5) < 1e-10
        assert metrics["classification"] == "medium"
        assert metrics["adequate"] is True


    def test_calculate_ductility_low(self, ensure_imports):
        """Calculate low ductility."""
        from pymemorial.visualization import calculate_ductility

        metrics = calculate_ductility(
            kappa_yield=0.003,
            kappa_ultimate=0.004,
        )
        
        # mu = 1.33
        assert metrics["mu"] < 1.5
        assert metrics["classification"] == "low"
        assert metrics["adequate"] is False

    def test_calculate_ductility_zero_yield(self, ensure_imports):
        """Calculate ductility with zero yield."""
        from pymemorial.visualization import calculate_ductility

        metrics = calculate_ductility(
            kappa_yield=0.0,
            kappa_ultimate=0.01,
        )
        
        assert metrics["mu"] == 0


@pytest.mark.unit
class TestCodeReference:
    """Test code reference formatting."""

    def test_format_code_reference_nbr(self, ensure_imports):
        """Format NBR code reference."""
        from pymemorial.visualization import format_code_reference, DesignCode

        ref = format_code_reference(DesignCode.NBR8800, "5.4.2")
        
        assert "NBR 8800" in ref
        assert "5.4.2" in ref
        assert "Item" in ref

    def test_format_code_reference_en(self, ensure_imports):
        """Format EN code reference."""
        from pymemorial.visualization import format_code_reference, DesignCode

        ref = format_code_reference(DesignCode.EN1993, "6.2.9")
        
        assert "EN 1993" in ref
        assert "6.2.9" in ref


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
