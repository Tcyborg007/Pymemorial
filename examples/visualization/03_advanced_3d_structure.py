# examples/visualization/03_advanced_3d_structure.py
"""
Example 3: Advanced 3D Structure Visualization.

Demonstrates:
- 3D truss/frame creation
- Multiple visualization engines
- Camera controls
- Professional lighting
- High-resolution export

Target: Advanced users
Time: 2 minutes
"""

from pathlib import Path
import numpy as np
from pymemorial.visualization import (
    create_visualizer,
    list_available_engines,
    PlotConfig,
    ExportConfig,
    ImageFormat,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Example 3: Advanced 3D Structure Visualization")
print("=" * 70)

# ============================================================================
# 1. CREATE 3D TRUSS STRUCTURE
# ============================================================================

# Define nodes (3D space frame)
nodes = np.array([
    # Base nodes (z=0)
    [0, 0, 0],
    [5, 0, 0],
    [5, 4, 0],
    [0, 4, 0],
    # Top nodes (z=3)
    [1, 1, 3],
    [4, 1, 3],
    [4, 3, 3],
    [1, 3, 3],
])

# Define members (element connectivity)
members = [
    # Base perimeter
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Top perimeter
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Vertical columns
    (0, 4), (1, 5), (2, 6), (3, 7),
    # Diagonal bracing
    (0, 5), (1, 4), (2, 7), (3, 6),
]

print(f"\n✓ Structure: {len(nodes)} nodes, {len(members)} members")

# ============================================================================
# 2. CHECK AVAILABLE 3D ENGINES
# ============================================================================

engines = list_available_engines()
print(f"✓ Available engines: {', '.join(engines)}")

has_3d = False
for engine_name in engines:
    viz = create_visualizer(engine=engine_name)
    if viz.supports_3d:
        print(f"  → {engine_name} supports 3D")
        has_3d = True

if not has_3d:
    print("⚠ No 3D-optimized engine available. Using default.")

# ============================================================================
# 3. CREATE 3D VISUALIZATION
# ============================================================================

viz = create_visualizer()

config = PlotConfig(
    title="3D Space Frame Structure",
    width=1200,
    height=900,
)

# Create 3D structure
fig = viz.create_3d_structure(nodes, members, config=config)

print("✓ 3D structure created")

# ============================================================================
# 4. EXPORT WITH DIFFERENT VIEWS
# ============================================================================

# Isometric view (default)
iso_path = viz.export_static(
    fig,
    ExportConfig(
        filename=OUTPUT_DIR / "structure_3d_isometric.png",
        format=ImageFormat.PNG,
        scale=2.0,
    ),
)
print(f"✓ Exported isometric view: {iso_path}")

# Interactive HTML
html_path = viz.export_static(
    fig,
    ExportConfig(
        filename=OUTPUT_DIR / "structure_3d_interactive.html",
        format=ImageFormat.HTML,
    ),
)
print(f"✓ Exported interactive HTML: {html_path}")
print("  → Open in browser to rotate, zoom, pan!")

# ============================================================================
# 5. TRY PYVISTA ENGINE IF AVAILABLE
# ============================================================================

try:
    from pymemorial.visualization import PyVistaEngine, PYVISTA_AVAILABLE
    
    if PYVISTA_AVAILABLE and PyVistaEngine is not None:
        print("\n✓ PyVista available! Creating advanced 3D render...")
        
        pv_engine = PyVistaEngine()
        
        # Create FEM mesh
        mesh = pv_engine.add_frame_members(
            nodes,
            members,
            radius=0.05,
            material="steel",
        )
        
        # Set camera preset
        pv_engine.set_camera_preset("isometric")
        
        # Add professional lighting
        pv_engine.add_professional_lighting()
        
        # Export high-quality screenshot
        pv_path = pv_engine.export_screenshot(
            OUTPUT_DIR / "structure_3d_pyvista.png",
            resolution=(1920, 1080),
            scale_factor=2,
        )
        
        print(f"  ✓ PyVista render: {pv_path}")
        
        # Close plotter
        pv_engine.close_plotter()
    else:
        print("\n⊘ PyVista not available (install with: pip install pymemorial[viz3d])")

except ImportError:
    print("\n⊘ PyVista not available (optional)")



    # ------------------------------------------------------------------------
    # ABSTRACT METHODS IMPLEMENTATION (required by VisualizerEngine ABC)
    # ------------------------------------------------------------------------

    def create_pm_diagram(
        self,
        p_values: NDArrayFloat,
        m_values: NDArrayFloat,
        design_point: Optional[Tuple[float, float]] = None,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create P-M interaction diagram (delegates to Plotly for 2D).

        PyVista is optimized for 3D visualization. For 2D diagrams,
        we delegate to PlotlyEngine.

        Args:
            p_values: Axial load values
            m_values: Moment values
            design_point: Optional (M, P) design point
            config: Plot configuration

        Returns:
            Plotly figure (not PyVista mesh)

        Note:
            This method exists to satisfy the ABC interface but
            PyVista is not ideal for 2D diagrams. Use PlotlyEngine instead.
        """
        logger.warning(
            "PyVistaEngine.create_pm_diagram() delegates to Plotly. "
            "Use create_visualizer(engine='plotly') for 2D diagrams."
        )

        # Fallback to Plotly
        from .plotly_engine import PlotlyEngine

        plotly = PlotlyEngine()
        return plotly.create_pm_diagram(p_values, m_values, design_point, config)

    def create_moment_curvature(
        self,
        curvature: NDArrayFloat,
        moment: NDArrayFloat,
        yield_point: Optional[Tuple[float, float]] = None,
        ultimate_point: Optional[Tuple[float, float]] = None,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create moment-curvature diagram (delegates to Plotly for 2D).

        See create_pm_diagram() docstring for rationale.
        """
        logger.warning(
            "PyVistaEngine.create_moment_curvature() delegates to Plotly. "
            "Use create_visualizer(engine='plotly') for 2D diagrams."
        )

        from .plotly_engine import PlotlyEngine

        plotly = PlotlyEngine()
        return plotly.create_moment_curvature(
            curvature, moment, yield_point, ultimate_point, config
        )

    def create_3d_structure(
        self,
        nodes: NDArrayFloat,
        elements: List[List[int]],
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create 3D structure visualization (native PyVista implementation).

        This is PyVista's strength - high-performance 3D rendering.

        Args:
            nodes: Node coordinates [N x 3]
            elements: Element connectivity
            config: Plot configuration

        Returns:
            PyVista mesh object

        Examples:
            >>> nodes = np.array([[0,0,0], [1,0,0], [1,1,0]])
            >>> elements = [[0,1], [1,2]]
            >>> mesh = pv_engine.create_3d_structure(nodes, elements)
        """
        self._check_available()

        # Create FEM mesh
        mesh = self.create_fem_mesh(nodes, elements, element_type="line")

        # Get plotter
        plotter = self._get_plotter()

        # Add mesh
        plotter.add_mesh(
            mesh,
            color="steelblue",
            line_width=3,
            render_lines_as_tubes=True,
            label="Structure",
        )

        # Add nodes
        node_cloud = pv.PolyData(nodes)
        plotter.add_mesh(
            node_cloud,
            color="red",
            point_size=10,
            render_points_as_spheres=True,
            label="Nodes",
        )

        # Set camera
        self.set_camera_preset("isometric")

        # Add axes
        self.add_axes()

        logger.info(f"Created 3D structure: {len(nodes)} nodes, {len(elements)} elements")

        return mesh

    def create_section_2d(
        self,
        vertices: NDArrayFloat,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create 2D section visualization (delegates to Plotly).

        PyVista is for 3D, so 2D sections are better in Plotly/Matplotlib.
        """
        logger.warning(
            "PyVistaEngine.create_section_2d() delegates to Plotly. "
            "Use create_visualizer(engine='plotly') for 2D sections."
        )

        from .plotly_engine import PlotlyEngine

        plotly = PlotlyEngine()
        return plotly.create_section_2d(vertices, config)

    def create_stress_contour(
        self,
        mesh: Any,
        stress_values: NDArrayFloat,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create stress contour visualization (native PyVista implementation).

        This is where PyVista excels - stress field visualization on 3D meshes.

        Args:
            mesh: PyVista mesh object
            stress_values: Stress values per node/element
            config: Plot configuration

        Returns:
            PyVista mesh with stress field

        Examples:
            >>> mesh = pv_engine.create_fem_mesh(nodes, elements)
            >>> stress = compute_von_mises(results)
            >>> pv_engine.create_stress_contour(mesh, stress)
        """
        self._check_available()

        # Add stress field
        self.add_stress_field(mesh, stress_values)

        return mesh




print("\n" + "=" * 70)
print("✓ Example 3 complete!")
print(f"  Files in: {OUTPUT_DIR}")
print("=" * 70)
