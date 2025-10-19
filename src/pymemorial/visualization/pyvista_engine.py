# src/pymemorial/visualization/pyvista_engine.py
"""
PyVista-based 3D FEM Visualization Engine for PyMemorial.

This module implements a production-grade 3D visualization engine using PyVista
for advanced finite element analysis (FEA) visualization, mesh rendering, and
interactive structural model exploration.

Features:
    - High-performance 3D mesh rendering (VTK backend)
    - Interactive camera controls and lighting
    - FEM-specific tools (stress fields, displacement contours)
    - Animation support (time-history, modal shapes)
    - Publication-quality screenshot export
    - VR/AR export capabilities (glTF, X3D)
    - GPU-accelerated volume rendering

PyVista is the Python interface to VTK (Visualization Toolkit), providing:
    - 10x-100x faster rendering than matplotlib 3D
    - True 3D camera manipulation
    - Support for massive meshes (>1M elements)
    - Client/server rendering for Jupyter notebooks

Requirements:
    - pyvista >= 0.44.0 (latest: 0.46.3, Oct 2025)
    - vtk >= 9.3.0 (bundled with PyVista)
    - numpy >= 1.24.0

Author: PyMemorial Team
License: MIT
Python: >=3.9

References:
    [1] PyVista Documentation: https://docs.pyvista.org/
    [2] VTK User Guide: https://vtk.org/documentation/
    [3] SciPy 2025 Tutorial: https://tutorial.pyvista.org/
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import base classes
from .base_visualizer import (
    DiagramType,
    ExportConfig,
    ImageFormat,
    NDArrayFloat,
    PlotConfig,
    Point3D,
    ThemeStyle,
    VisualizerEngine,
)

# ============================================================================
# OPTIONAL IMPORTS - Graceful degradation
# ============================================================================

logger = logging.getLogger(__name__)

# Try importing PyVista
PYVISTA_AVAILABLE = False
PYVISTA_VERSION = "0.0.0"
VTK_VERSION = "0.0.0"

try:
    import pyvista as pv
    from pyvista import examples

    PYVISTA_AVAILABLE = True
    PYVISTA_VERSION = pv.__version__
    
    # Get VTK version
    try:
        import vtk
        VTK_VERSION = vtk.vtkVersion.GetVTKVersion()
    except:
        VTK_VERSION = "unknown"
    
    logger.info(f"PyVista {PYVISTA_VERSION} loaded successfully (VTK {VTK_VERSION})")
    
    # Configure PyVista defaults for best performance
    pv.set_plot_theme("document")  # Clean theme
    pv.global_theme.window_size = [1024, 768]  # Default window size
    pv.global_theme.show_edges = False  # Smooth rendering
    pv.global_theme.split_sharp_edges = True  # Better normals
    
except ImportError as e:
    logger.warning(f"PyVista not available: {e}")
    logger.warning("Install with: pip install 'pymemorial[viz3d]'")


# ============================================================================
# CONSTANTS - Colormaps and styling
# ============================================================================

# PyVista-compatible colormaps (Matplotlib + VTK native)
STRESS_COLORMAPS: Dict[str, str] = {
    "von_mises": "turbo",  # Modern colormap
    "principal_stress": "RdYlGn_r",  # Red = high stress
    "displacement": "viridis",
    "strain": "plasma",
    "temperature": "inferno",
    "velocity": "coolwarm",
    "seismic": "seismic",  # Blue-white-red
}

# Material visualization styles
MATERIAL_RENDER_STYLES: Dict[str, Dict[str, Any]] = {
    "steel": {
        "color": "#B0C4DE",
        "metallic": 0.8,
        "roughness": 0.3,
        "show_edges": True,
        "edge_color": "#2F4F4F",
    },
    "concrete": {
        "color": "#D3D3D3",
        "metallic": 0.0,
        "roughness": 0.9,
        "show_edges": False,
    },
    "wood": {
        "color": "#DEB887",
        "metallic": 0.0,
        "roughness": 0.7,
        "show_edges": True,
    },
    "aluminum": {
        "color": "#E8E8E8",
        "metallic": 0.9,
        "roughness": 0.2,
        "show_edges": False,
    },
}

# Camera presets for structural views
CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    "isometric": {"position": (1, 1, 1), "viewup": (0, 0, 1)},
    "top": {"position": (0, 0, 1), "viewup": (0, 1, 0)},
    "front": {"position": (0, -1, 0), "viewup": (0, 0, 1)},
    "side": {"position": (1, 0, 0), "viewup": (0, 0, 1)},
    "elevation": {"position": (0, -1, 0.5), "viewup": (0, 0, 1)},
}


# ============================================================================
# PYVISTA ENGINE CLASS
# ============================================================================


class PyVistaEngine(VisualizerEngine):
    """
    PyVista-based 3D visualization engine for advanced FEM rendering.

    This engine provides high-performance 3D visualization using VTK through
    PyVista's Pythonic API. Ideal for interactive mesh visualization, stress
    field rendering, and publication-quality structural models.

    Features:
        - Interactive 3D rendering with camera controls
        - GPU-accelerated volume rendering
        - Support for massive meshes (>1M elements)
        - Stress/displacement field visualization
        - Animation capabilities (modal shapes, time-history)
        - Export to VTK, STL, glTF, X3D formats
        - Jupyter notebook integration (trame backend)

    Attributes:
        name: Engine identifier ("pyvista")
        version: PyVista library version
        supports_3d: True (native 3D engine)
        supports_interactive: True (interactive by default)
        available: Whether PyVista is installed
        plotter: Current PyVista plotter instance

    Examples:
        >>> engine = PyVistaEngine()
        >>> if engine.available:
        ...     mesh = engine.create_fem_mesh(nodes, elements)
        ...     engine.add_stress_field(mesh, stress_values)
        ...     engine.show(interactive=True)
        ... else:
        ...     print("Install: pip install pymemorial[viz3d]")

    Performance Notes:
        - For meshes >100k elements, use `render_lines_as_tubes=False`
        - Enable GPU rendering with `enable_gpu_acceleration()`
        - Use `decimation()` for preview rendering of massive meshes

    References:
        [1] PyVista Best Practices: https://docs.pyvista.org/user-guide/
        [2] VTK Performance Guide: https://vtk.org/Wiki/VTK/Performance
    """

    def __init__(self) -> None:
        """Initialize PyVista visualization engine."""
        super().__init__(name="pyvista")

        if not PYVISTA_AVAILABLE:
            logger.error("PyVista not available - engine disabled")
            return

        self._plotter: Optional[pv.Plotter] = None
        self._render_settings = {}  # PyVista 0.46+ removed these params
        self._performance_mode = False
        self._use_gpu = True
        self._off_screen_mode = False  # Track current mode


    # ------------------------------------------------------------------------
    # PROPERTIES (implementing ABC)
    # ------------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Return PyVista version."""
        return PYVISTA_VERSION

    @property
    def supports_3d(self) -> bool:
        """PyVista is a native 3D engine."""
        return True

    @property
    def supports_interactive(self) -> bool:
        """PyVista supports interactive rendering."""
        return True

    @property
    def available(self) -> bool:
        """Check if PyVista is installed and functional."""
        return PYVISTA_AVAILABLE

    @property
    def supported_formats(self) -> List[ImageFormat]:
        """Return supported export formats."""
        if not PYVISTA_AVAILABLE:
            return []

        # PyVista supports many formats
        return [
            ImageFormat.PNG,
            ImageFormat.JPEG,
            ImageFormat.SVG,
            ImageFormat.PDF,
            ImageFormat.EPS,
            # 3D formats (via file extension)
            # VTK, STL, PLY, glTF, X3D, OBJ
        ]

    # ------------------------------------------------------------------------
    # PLOTTER MANAGEMENT
    # ------------------------------------------------------------------------

    def _get_plotter(
        self,
        off_screen: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
    ) -> pv.Plotter:
        """
        Get or create PyVista plotter instance.

        IMPORTANT: If off_screen mode changes, must create new plotter.

        Args:
            off_screen: Render without displaying window
            window_size: (width, height) in pixels

        Returns:
            PyVista Plotter object
        """
        if not self.available:
            raise RuntimeError("PyVista engine not available")

        # Must recreate plotter if off_screen mode changed
        needs_new = (
            self._plotter is None
            or self._plotter._closed
            or (self._off_screen_mode != off_screen)
        )

        if needs_new:
            # Close old plotter
            if self._plotter is not None and not self._plotter._closed:
                try:
                    self._plotter.close()
                except:
                    pass

            size = window_size or (1024, 768)

            # Create new plotter with correct mode
            self._plotter = pv.Plotter(
                off_screen=off_screen,
                window_size=size,
            )

            self._off_screen_mode = off_screen
            self._plotter.set_background("white")

            try:
                self._plotter.enable_anti_aliasing("msaa")
            except:
                pass

            logger.debug(f"Created PyVista plotter: {size}, off_screen={off_screen}")

        return self._plotter



    def close_plotter(self) -> None:
        """Close current plotter instance."""
        if self._plotter is not None and not self._plotter._closed:
            self._plotter.close()
            self._plotter = None
            logger.debug("Closed PyVista plotter")

    # ------------------------------------------------------------------------
    # MESH CREATION METHODS
    # ------------------------------------------------------------------------

    def create_fem_mesh(
        self,
        nodes: NDArrayFloat,
        elements: List[List[int]],
        element_type: Literal["line", "triangle", "quad", "tetra", "hexa"] = "line",
    ) -> pv.PolyData | pv.UnstructuredGrid:
        """
        Create FEM mesh from nodes and element connectivity.

        Args:
            nodes: Node coordinates [N x 3] in meters
            elements: Element connectivity [[node_ids...], ...]
            element_type: Element geometry type

        Returns:
            PyVista mesh object

        Examples:
            >>> nodes = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
            >>> elements = [[0,1,2,3]]  # Quad
            >>> mesh = engine.create_fem_mesh(nodes, elements, "quad")
        """
        self._check_available()

        if nodes.shape[1] != 3:
            raise ValueError(f"Nodes must be N x 3, got {nodes.shape}")

        # Map element type to VTK cell type
        vtk_cell_types = {
            "line": 3,  # VTK_LINE
            "triangle": 5,  # VTK_TRIANGLE
            "quad": 9,  # VTK_QUAD
            "tetra": 10,  # VTK_TETRA
            "hexa": 12,  # VTK_HEXAHEDRON
        }

        if element_type not in vtk_cell_types:
            raise ValueError(f"Unknown element_type: {element_type}")

        # For simple line elements, use PolyData (faster)
        if element_type == "line":
            # Pad edges for VTK
            n_points_per_edge = 2
            edges = []
            for elem in elements:
                edges.append([n_points_per_edge] + list(elem))
            
            mesh = pv.PolyData(nodes, edges)
            
        else:
            # For 2D/3D elements, use UnstructuredGrid
            cell_type = vtk_cell_types[element_type]
            cells = []
            
            for elem in elements:
                cells.append([len(elem)] + list(elem))
            
            cells_array = np.hstack(cells).astype(np.int32)
            cell_types = np.full(len(elements), cell_type, dtype=np.uint8)
            
            mesh = pv.UnstructuredGrid(cells_array, cell_types, nodes)

        logger.info(f"Created FEM mesh: {len(nodes)} nodes, {len(elements)} elements ({element_type})")
        
        return mesh

    # Continua na PARTE 2...
    # ------------------------------------------------------------------------
    # VISUALIZATION METHODS - Stress Fields & Contours
    # ------------------------------------------------------------------------

    def add_stress_field(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        stress_values: NDArrayFloat,
        stress_type: Literal["von_mises", "principal_1", "principal_3"] = "von_mises",
        scalar_name: str = "Stress",
        show_scalar_bar: bool = True,
        clim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Add stress field visualization to mesh.

        Args:
            mesh: PyVista mesh object
            stress_values: Stress values per node/cell (Pa)
            stress_type: Type of stress to visualize
            scalar_name: Name for scalar field
            show_scalar_bar: Show colorbar legend
            clim: (min, max) stress range for colormap

        Examples:
            >>> mesh = engine.create_fem_mesh(nodes, elements)
            >>> stress = compute_von_mises_stress(fem_results)
            >>> engine.add_stress_field(mesh, stress, show_scalar_bar=True)
        """
        self._check_available()

        # Convert stress to MPa for better readability
        stress_mpa = stress_values / 1e6

        # Add as point or cell data
        if len(stress_mpa) == mesh.n_points:
            mesh.point_data[scalar_name] = stress_mpa
        elif len(stress_mpa) == mesh.n_cells:
            mesh.cell_data[scalar_name] = stress_mpa
        else:
            raise ValueError(
                f"Stress array length {len(stress_mpa)} doesn't match "
                f"n_points {mesh.n_points} or n_cells {mesh.n_cells}"
            )

        # Get plotter
        plotter = self._get_plotter()

        # Select colormap
        cmap = STRESS_COLORMAPS.get(stress_type, "turbo")

        # Determine color limits
        if clim is None:
            clim = (stress_mpa.min(), stress_mpa.max())

        # Add mesh with stress coloring
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": f"{stress_type.replace('_', ' ').title()} Stress (MPa)",
                "vertical": True,
                "height": 0.7,
                "position_x": 0.85,
                "position_y": 0.15,
                "title_font_size": 16,
                "label_font_size": 12,
                "n_labels": 5,
                "fmt": "%.1f",
            },
            render_lines_as_tubes=False,  # Better performance
            smooth_shading=True,
        )

        logger.info(f"Added stress field: {stress_type}, range: {clim[0]:.1f}-{clim[1]:.1f} MPa")

    def add_displacement_field(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        displacements: NDArrayFloat,
        scale_factor: float = 1.0,
        show_undeformed: bool = True,
        warp_by_vector: bool = True,
    ) -> pv.PolyData | pv.UnstructuredGrid:
        """
        Add displacement field visualization.

        Args:
            mesh: Original (undeformed) mesh
            displacements: Displacement vectors [N x 3] in meters
            scale_factor: Magnification factor for displacements
            show_undeformed: Show original mesh as wireframe
            warp_by_vector: Warp mesh geometry by displacement

        Returns:
            Warped mesh object

        Examples:
            >>> mesh = engine.create_fem_mesh(nodes, elements)
            >>> displacements = fem_results["nodal_displacements"]
            >>> warped = engine.add_displacement_field(mesh, displacements, scale_factor=10)
        """
        self._check_available()

        if displacements.shape != (mesh.n_points, 3):
            raise ValueError(
                f"Displacements shape {displacements.shape} doesn't match "
                f"mesh points ({mesh.n_points}, 3)"
            )

        # Add displacement vectors as point data
        mesh.point_data["Displacement"] = displacements

        # Compute displacement magnitude
        displacement_mag = np.linalg.norm(displacements, axis=1)
        mesh.point_data["Displacement_Magnitude"] = displacement_mag

        plotter = self._get_plotter()

        # Show undeformed mesh as wireframe
        if show_undeformed:
            plotter.add_mesh(
                mesh,
                color="gray",
                style="wireframe",
                line_width=1,
                opacity=0.3,
                label="Undeformed",
            )

        # Warp mesh by displacement vectors
        if warp_by_vector:
            warped_mesh = mesh.warp_by_vector(
                vectors="Displacement", factor=scale_factor
            )

            # Add warped mesh with displacement magnitude coloring
            plotter.add_mesh(
                warped_mesh,
                scalars="Displacement_Magnitude",
                cmap="viridis",
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": f"Displacement (mm) [×{scale_factor}]",
                    "vertical": True,
                },
                smooth_shading=True,
                label="Deformed",
            )

            logger.info(
                f"Added displacement field: scale={scale_factor}, "
                f"max_disp={displacement_mag.max()*1000:.2f} mm"
            )

            return warped_mesh
        else:
            # Just color by magnitude without warping
            plotter.add_mesh(
                mesh,
                scalars="Displacement_Magnitude",
                cmap="viridis",
                show_scalar_bar=True,
            )
            return mesh

    def add_vector_field(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        vectors: NDArrayFloat,
        vector_name: str = "Vectors",
        glyph_type: Literal["arrow", "cone", "sphere"] = "arrow",
        scale_factor: float = 0.1,
        color: Optional[str] = None,
    ) -> None:
        """
        Add vector field visualization (loads, velocities, etc.).

        Args:
            mesh: Mesh to attach vectors to
            vectors: Vector field [N x 3]
            vector_name: Name for vector field
            glyph_type: Glyph geometry for vectors
            scale_factor: Size of glyphs
            color: Optional fixed color for glyphs

        Examples:
            >>> mesh = engine.create_fem_mesh(nodes, elements)
            >>> loads = np.array([[0, 0, -100], [0, 0, -50], ...])  # kN
            >>> engine.add_vector_field(mesh, loads, "Applied Loads", glyph_type="arrow")
        """
        self._check_available()

        if vectors.shape != (mesh.n_points, 3):
            raise ValueError(f"Vectors shape {vectors.shape} must be (n_points, 3)")

        # Add vectors to mesh
        mesh.point_data[vector_name] = vectors

        # Compute magnitude
        magnitude = np.linalg.norm(vectors, axis=1)
        mesh.point_data[f"{vector_name}_Magnitude"] = magnitude

        plotter = self._get_plotter()

        # Create glyphs
        if glyph_type == "arrow":
            glyph = pv.Arrow(scale=scale_factor)
        elif glyph_type == "cone":
            glyph = pv.Cone(height=scale_factor, radius=scale_factor * 0.3)
        elif glyph_type == "sphere":
            glyph = pv.Sphere(radius=scale_factor)
        else:
            raise ValueError(f"Unknown glyph_type: {glyph_type}")

        # Generate glyphs oriented by vectors
        glyphs = mesh.glyph(
            orient=vector_name,
            scale=f"{vector_name}_Magnitude",
            factor=scale_factor,
            geom=glyph,
        )

        # Add glyphs to scene
        if color:
            plotter.add_mesh(glyphs, color=color, label=vector_name)
        else:
            plotter.add_mesh(
                glyphs,
                scalars=f"{vector_name}_Magnitude",
                cmap="Reds",
                show_scalar_bar=False,
                label=vector_name,
            )

        logger.info(f"Added vector field: {vector_name}, {len(vectors)} vectors")

    # ------------------------------------------------------------------------
    # STRUCTURAL ELEMENT RENDERING
    # ------------------------------------------------------------------------

    def add_frame_members(
        self,
        nodes: NDArrayFloat,
        members: List[Tuple[int, int]],
        radius: float = 0.05,
        material: str = "steel",
        show_nodes: bool = True,
    ) -> pv.PolyData:
        """
        Add frame members (beams, columns) as solid tubes.

        Args:
            nodes: Node coordinates [N x 3] in meters
            members: Member connectivity [(i, j), ...]
            radius: Member radius in meters
            material: Material name for styling
            show_nodes: Show nodes as spheres

        Returns:
            Combined mesh of all members

        Examples:
            >>> nodes = np.array([[0,0,0], [5,0,0], [5,0,3], [0,0,3]])
            >>> members = [(0,1), (1,2), (2,3), (3,0)]
            >>> mesh = engine.add_frame_members(nodes, members, radius=0.1)
        """
        self._check_available()

        # Force float32 to avoid PyVista warnings
        nodes = np.asarray(nodes, dtype=np.float32)

        if nodes.shape[1] != 3:
            raise ValueError(f"Nodes must be N x 3, got {nodes.shape}")

        # Get plotter (NOT off_screen by default for interactive work)
        plotter = self._get_plotter(off_screen=False)

        # Get material styling
        style = MATERIAL_RENDER_STYLES.get(material, MATERIAL_RENDER_STYLES["steel"])

        # Create tubes for each member
        all_tubes = []

        for i, j in members:
            # Create line between nodes
            line = pv.Line(nodes[i], nodes[j])

            # Convert to tube
            tube = line.tube(radius=radius, n_sides=20)
            all_tubes.append(tube)

        # Merge all tubes
        combined_mesh = all_tubes[0]
        for tube in all_tubes[1:]:
            combined_mesh = combined_mesh.merge(tube)

        # Add to plotter with material styling
        plotter.add_mesh(
            combined_mesh,
            color=style["color"],
            show_edges=style["show_edges"],
            edge_color=style.get("edge_color", "black"),
            smooth_shading=True,
            metallic=style.get("metallic", 0.5),
            roughness=style.get("roughness", 0.5),
            label=material.capitalize(),
        )

        # Add nodes as spheres
        if show_nodes:
            node_cloud = pv.PolyData(nodes)
            spheres = node_cloud.glyph(geom=pv.Sphere(radius=radius * 1.5), scale=False)

            plotter.add_mesh(
                spheres,
                color="black",
                label="Nodes",
            )

        logger.info(f"Added {len(members)} frame members ({material})")

        return combined_mesh


    def add_shell_elements(
        self,
        nodes: NDArrayFloat,
        shells: List[List[int]],
        thickness: Optional[NDArrayFloat] = None,
        material: str = "concrete",
    ) -> pv.UnstructuredGrid:
        """
        Add shell/plate elements (floors, walls, slabs).

        Args:
            nodes: Node coordinates [N x 3]
            shells: Shell connectivity [[n1, n2, n3, n4], ...]
            thickness: Optional thickness per shell (for extrusion)
            material: Material for styling

        Returns:
            Shell mesh

        Examples:
            >>> # Floor slab
            >>> nodes = np.array([[0,0,0], [10,0,0], [10,5,0], [0,5,0]])
            >>> shells = [[0,1,2,3]]
            >>> mesh = engine.add_shell_elements(nodes, shells, material="concrete")
        """
        self._check_available()

        # Determine element type (triangle or quad)
        n_nodes_per_elem = len(shells[0])
        if n_nodes_per_elem == 3:
            element_type = "triangle"
        elif n_nodes_per_elem == 4:
            element_type = "quad"
        else:
            raise ValueError(f"Shells must have 3 or 4 nodes, got {n_nodes_per_elem}")

        # Create mesh
        mesh = self.create_fem_mesh(nodes, shells, element_type)

        # Get material styling
        style = MATERIAL_RENDER_STYLES.get(material, MATERIAL_RENDER_STYLES["concrete"])

        plotter = self._get_plotter()

        # Add mesh
        plotter.add_mesh(
            mesh,
            color=style["color"],
            show_edges=style["show_edges"],
            smooth_shading=True,
            opacity=0.9,  # Slight transparency for shells
            label=f"{material.capitalize()} Shell",
        )

        logger.info(f"Added {len(shells)} shell elements ({material})")

        return mesh

    # ------------------------------------------------------------------------
    # CAMERA AND LIGHTING CONTROLS
    # ------------------------------------------------------------------------

    def set_camera_preset(
        self,
        preset: Literal["isometric", "top", "front", "side", "elevation"] = "isometric"
    ) -> None:
        """
        Set camera to predefined view.

        Args:
            preset: Camera preset name

        Examples:
            >>> engine.set_camera_preset("isometric")
            >>> engine.set_camera_preset("elevation")
        """
        self._check_available()

        if preset not in CAMERA_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(CAMERA_PRESETS.keys())}")

        plotter = self._get_plotter()
        camera_params = CAMERA_PRESETS[preset]

        plotter.camera_position = [
            camera_params["position"],
            (0, 0, 0),  # Focal point (origin)
            camera_params["viewup"],
        ]

        logger.debug(f"Camera set to: {preset}")

    def add_professional_lighting(self) -> None:
        """
        Add studio-quality lighting for publication renders.

        Uses three-point lighting: key light, fill light, rim light.
        """
        self._check_available()

        plotter = self._get_plotter()

        # Remove default lights
        plotter.remove_all_lights()

        # Key light (main, bright)
        key_light = pv.Light(position=(10, 10, 10), intensity=0.8)
        plotter.add_light(key_light)

        # Fill light (soften shadows)
        fill_light = pv.Light(position=(-10, 5, 5), intensity=0.3)
        plotter.add_light(fill_light)

        # Rim light (edge highlight)
        rim_light = pv.Light(position=(0, -10, 8), intensity=0.4)
        plotter.add_light(rim_light)

        # Ambient light (overall brightness)
        plotter.add_light(pv.Light(light_type="headlight", intensity=0.2))

        logger.info("Professional lighting added")

    def enable_shadows(self, enable: bool = True) -> None:
        """
        Enable shadow rendering (computationally expensive).

        Args:
            enable: Enable/disable shadows
        """
        self._check_available()

        plotter = self._get_plotter()
        plotter.enable_shadows() if enable else plotter.disable_shadows()

        logger.info(f"Shadows {'enabled' if enable else 'disabled'}")

    # ------------------------------------------------------------------------
    # ANNOTATIONS AND LABELS
    # ------------------------------------------------------------------------

    def add_axes(
        self,
        position: Literal["lower_left", "lower_right", "upper_left", "upper_right"] = "lower_left",
        interactive: bool = False,
    ) -> None:
        """
        Add coordinate axes indicator.

        Args:
            position: Corner position for axes
            interactive: Allow user to click axes for view alignment
        """
        self._check_available()

        plotter = self._get_plotter()

        plotter.add_axes(
            interactive=interactive,
            color="black",
            x_color="red",
            y_color="green",
            z_color="blue",
        )

        # Add axes widget in corner
        plotter.add_axes_at_origin(labels_off=False)

        logger.debug(f"Axes added at {position}")

    def add_scale_bar(
        self,
        length: float = 1.0,
        units: str = "m",
        position: Tuple[float, float] = (0.1, 0.05),
    ) -> None:
        """
        Add scale bar for dimensional reference.

        Args:
            length: Physical length of scale bar
            units: Unit label
            position: (x, y) in normalized viewport coordinates
        """
        self._check_available()

        plotter = self._get_plotter()

        # PyVista doesn't have built-in scale bar, so add text annotation
        plotter.add_text(
            f"Scale: {length} {units}",
            position=position,
            font_size=12,
            color="black",
        )

        logger.debug(f"Scale bar added: {length} {units}")

    # Continua na PARTE 3 (export, animation, performance)...
    # ------------------------------------------------------------------------
    # EXPORT METHODS - High-quality screenshots & 3D formats
    # ------------------------------------------------------------------------

    def export_screenshot(
        self,
        filepath: Path | str,
        resolution: Tuple[int, int] = (1920, 1080),
        transparent_background: bool = False,
        scale_factor: int = 1,
    ) -> Path:
        """
        Export high-resolution screenshot.

        This method uses the show() trick with screenshot parameter,
        which is the recommended PyVista approach for headless screenshots.

        Args:
            filepath: Output file path
            resolution: (width, height) in pixels
            transparent_background: Transparent PNG background
            scale_factor: Super-sampling multiplier (1-4)

        Returns:
            Path to saved image
        """
        self._check_available()

        filepath = Path(filepath)

        # Check if we have content to render
        if self._plotter is None or self._plotter._closed:
            raise RuntimeError(
                "No plotter content. Call add_frame_members() or "
                "other visualization methods first."
            )

        # Get current plotter
        plotter = self._plotter

        # Configure for export
        if transparent_background:
            plotter.set_background("white", top="white")

        plotter.window_size = resolution

        # PyVista recommended approach: use show() with screenshot param
        # This works for both on-screen and off-screen plotters
        try:
            plotter.show(
                auto_close=False,  # Don't close after show
                interactive=False,  # No interaction needed
                screenshot=str(filepath),  # Save screenshot
            )
        except Exception as e:
            logger.error(f"Screenshot via show() failed: {e}")

            # Fallback: try direct screenshot after render
            try:
                plotter.render()
                plotter.screenshot(
                    filename=str(filepath),
                    transparent_background=transparent_background,
                    scale=scale_factor,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Both screenshot methods failed. "
                    f"Show: {e}, Screenshot: {e2}"
                )

        logger.info(
            f"Exported screenshot: {filepath} ({resolution[0]}x{resolution[1]}, scale={scale_factor})"
        )

        return filepath



    def export_3d_model(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        filepath: Path | str,
        format: Literal["vtk", "stl", "ply", "obj", "gltf"] = "stl",
        binary: bool = True,
    ) -> Path:
        """
        Export mesh to 3D file format.

        Args:
            mesh: Mesh to export
            filepath: Output file path
            format: File format
            binary: Use binary encoding (smaller files)

        Returns:
            Path to saved file

        Examples:
            >>> mesh = engine.create_fem_mesh(nodes, elements)
            >>> engine.export_3d_model(mesh, "structure.stl")
            >>> engine.export_3d_model(mesh, "model.gltf")  # For AR/VR
        """
        self._check_available()

        filepath = Path(filepath)

        # Ensure correct extension
        if not filepath.suffix:
            filepath = filepath.with_suffix(f".{format}")

        # Export based on format
        if format == "vtk":
            mesh.save(str(filepath), binary=binary)
        elif format == "stl":
            mesh.save(str(filepath), binary=binary)
        elif format == "ply":
            mesh.save(str(filepath), binary=binary)
        elif format == "obj":
            # OBJ is always ASCII
            mesh.save(str(filepath))
        elif format == "gltf":
            # glTF for AR/VR applications
            if not str(filepath).endswith((".gltf", ".glb")):
                filepath = filepath.with_suffix(".gltf")
            mesh.save(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported 3D model: {filepath} ({format})")

        return filepath

    def export_static(
        self, figure: Any, config: ExportConfig
    ) -> Path:
        """
        Export static image (implementing ABC interface).

        Args:
            figure: PyVista plotter or mesh
            config: Export configuration

        Returns:
            Path to exported file
        """
        self._check_available()

        # If figure is a mesh, render it first
        if isinstance(figure, (pv.PolyData, pv.UnstructuredGrid)):
            plotter = self._get_plotter(off_screen=True)
            plotter.add_mesh(figure)
        else:
            plotter = self._plotter

        # Export based on format
        if config.format == ImageFormat.PNG:
            return self.export_screenshot(
                config.corrected_filename,
                transparent_background=config.transparent_bg,
                scale_factor=int(config.scale),
            )
        elif config.format in [ImageFormat.JPEG, ImageFormat.SVG, ImageFormat.PDF]:
            # PyVista supports these via matplotlib
            return self.export_screenshot(
                config.corrected_filename,
                scale_factor=int(config.scale),
            )
        else:
            raise ValueError(f"Unsupported export format: {config.format}")

    # ------------------------------------------------------------------------
    # ANIMATION METHODS - Time-history and modal shapes
    # ------------------------------------------------------------------------

    def create_animation(
        self,
        meshes: List[pv.PolyData | pv.UnstructuredGrid],
        output_path: Path | str,
        fps: int = 24,
        n_frames: Optional[int] = None,
        loop: bool = True,
    ) -> Path:
        """
        Create animation from mesh sequence.

        Args:
            meshes: List of meshes (one per frame)
            output_path: Output video/GIF path
            fps: Frames per second
            n_frames: Number of frames (None = all meshes)
            loop: Loop animation

        Returns:
            Path to animation file

        Examples:
            >>> # Modal shape animation
            >>> meshes = []
            >>> for t in np.linspace(0, 2*np.pi, 60):
            ...     displacement = mode_shape * np.sin(t)
            ...     warped = mesh.warp_by_vector(displacement)
            ...     meshes.append(warped)
            >>> engine.create_animation(meshes, "modal_shape.gif", fps=30)
        """
        self._check_available()

        output_path = Path(output_path)

        if n_frames is None:
            n_frames = len(meshes)

        # Create plotter for animation
        plotter = self._get_plotter(off_screen=True)

        # Set up recording
        plotter.open_gif(str(output_path))

        # Render each frame
        for i, mesh in enumerate(meshes[:n_frames]):
            plotter.clear()
            plotter.add_mesh(mesh, smooth_shading=True)
            plotter.write_frame()

        plotter.close()

        logger.info(f"Created animation: {output_path} ({n_frames} frames @ {fps} fps)")

        return output_path

    def animate_modal_shape(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        mode_shape: NDArrayFloat,
        output_path: Path | str,
        n_cycles: int = 2,
        fps: int = 30,
        scale_factor: float = 1.0,
    ) -> Path:
        """
        Animate structural modal shape.

        Args:
            mesh: Base mesh
            mode_shape: Mode shape vectors [N x 3]
            output_path: Output animation path
            n_cycles: Number of oscillation cycles
            fps: Frames per second
            scale_factor: Amplitude scaling

        Returns:
            Path to animation

        Examples:
            >>> mesh = engine.create_fem_mesh(nodes, elements)
            >>> mode1 = fem_results["mode_shapes"][0]  # First mode
            >>> engine.animate_modal_shape(mesh, mode1, "mode1.gif", scale_factor=5)
        """
        self._check_available()

        # Generate frames
        n_frames = n_cycles * fps
        time = np.linspace(0, n_cycles * 2 * np.pi, n_frames)

        meshes = []
        for t in time:
            # Sinusoidal displacement
            displacement = mode_shape * np.sin(t) * scale_factor

            # Add to mesh and warp
            mesh_copy = mesh.copy()
            mesh_copy.point_data["Mode"] = displacement
            warped = mesh_copy.warp_by_vector(vectors="Mode", factor=1.0)

            meshes.append(warped)

        # Create animation
        return self.create_animation(meshes, output_path, fps=fps)

    # ------------------------------------------------------------------------
    # PERFORMANCE OPTIMIZATION
    # ------------------------------------------------------------------------

    def enable_gpu_acceleration(self) -> None:
        """
        Enable GPU-accelerated rendering (requires compatible GPU).

        Uses VTK's OpenGL backend for hardware acceleration.
        """
        self._check_available()

        # Already enabled by default in PyVista
        self._use_gpu = True

        logger.info("GPU acceleration enabled")

    def enable_performance_mode(self, enable: bool = True) -> None:
        """
        Toggle performance mode (lower quality, faster rendering).

        Useful for interactive manipulation of large meshes.

        Args:
            enable: Enable/disable performance mode
        """
        self._check_available()

        self._performance_mode = enable

        if enable:
            # Reduce quality for speed
            self._render_settings["multi_samples"] = 0  # Disable MSAA
            self._render_settings["line_smoothing"] = False
            self._render_settings["polygon_smoothing"] = False
            logger.info("Performance mode ENABLED (lower quality)")
        else:
            # Restore quality
            self._render_settings["multi_samples"] = 8
            self._render_settings["line_smoothing"] = True
            self._render_settings["polygon_smoothing"] = True
            logger.info("Performance mode DISABLED (high quality)")

    def decimate_mesh(
        self,
        mesh: pv.PolyData | pv.UnstructuredGrid,
        target_reduction: float = 0.5,
    ) -> pv.PolyData:
        """
        Reduce mesh complexity for faster rendering.

        Args:
            mesh: Original mesh
            target_reduction: Fraction to reduce (0.5 = 50% reduction)

        Returns:
            Decimated mesh

        Examples:
            >>> large_mesh = engine.create_fem_mesh(nodes, elements)
            >>> preview_mesh = engine.decimate_mesh(large_mesh, target_reduction=0.9)
            >>> # 90% reduction for quick preview
        """
        self._check_available()

        decimated = mesh.decimate(target_reduction)

        logger.info(
            f"Decimated mesh: {mesh.n_cells} → {decimated.n_cells} cells "
            f"({target_reduction*100:.0f}% reduction)"
        )

        return decimated

    # ------------------------------------------------------------------------
    # SHOW METHODS - Display & Interaction
    # ------------------------------------------------------------------------

    def show(
        self,
        interactive: bool = True,
        full_screen: bool = False,
        screenshot: Optional[Path | str] = None,
    ) -> None:
        """
        Display current scene.

        Args:
            interactive: Enable interactive controls
            full_screen: Full screen mode
            screenshot: Save screenshot to path

        Examples:
            >>> engine.show(interactive=True)
            >>> # User can rotate, zoom, pan with mouse
        """
        self._check_available()

        plotter = self._get_plotter()

        # Configure interaction
        if not interactive:
            plotter.enable_parallel_projection()

        # Full screen
        if full_screen:
            plotter.full_screen()

        # Show
        plotter.show(
            interactive=interactive,
            auto_close=True,
            screenshot=str(screenshot) if screenshot else None,
        )

        logger.info(f"Displayed scene (interactive={interactive})")

    def show_jupyter(self) -> Any:
        """
        Show in Jupyter notebook (requires trame backend).

        Returns:
            Widget for embedding in notebook

        Examples:
            >>> # In Jupyter notebook:
            >>> widget = engine.show_jupyter()
            >>> widget  # Display in cell
        """
        self._check_available()

        try:
            import trame  # noqa: F401
        except ImportError:
            logger.warning("Trame not installed. Install with: pip install trame")
            return None

        plotter = self._get_plotter()

        # Return widget for Jupyter
        return plotter.show(
            jupyter_backend="trame",
            return_viewer=True,
            auto_close=False,
        )

    # ------------------------------------------------------------------------
    # UTILITY METHODS
    # ------------------------------------------------------------------------

    def _check_available(self) -> None:
        """Raise error if engine not available."""
        if not self.available:
            raise RuntimeError(
                "PyVistaEngine not available. Install with: "
                "pip install 'pymemorial[viz3d]'"
            )

    def get_mesh_info(
        self, mesh: pv.PolyData | pv.UnstructuredGrid
    ) -> Dict[str, Any]:
        """
        Get detailed mesh information.

        Args:
            mesh: Mesh to analyze

        Returns:
            Dictionary with mesh statistics
        """
        return {
            "n_points": mesh.n_points,
            "n_cells": mesh.n_cells,
            "bounds": mesh.bounds,
            "center": mesh.center,
            "volume": mesh.volume if hasattr(mesh, "volume") else None,
            "point_data_arrays": list(mesh.point_data.keys()),
            "cell_data_arrays": list(mesh.cell_data.keys()),
        }

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
        """
        logger.warning(
            "PyVistaEngine.create_pm_diagram() delegates to Plotly. "
            "Use create_visualizer(engine='plotly') for 2D diagrams."
        )

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
        """Create moment-curvature diagram (delegates to Plotly for 2D)."""
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
        """Create 3D structure visualization (native PyVista implementation)."""
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
        """Create 2D section visualization (delegates to Plotly)."""
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
        """Create stress contour visualization (native PyVista implementation)."""
        self._check_available()

        # Add stress field
        self.add_stress_field(mesh, stress_values)

        return mesh


    def __repr__(self) -> str:
        """Return string representation."""
        status = "available" if self.available else "not available"
        return (
            f"<PyVistaEngine v{self.version} ({status})>\n"
            f"  VTK: {VTK_VERSION}\n"
            f"  3D: {self.supports_3d}\n"
            f"  Interactive: {self.supports_interactive}\n"
            f"  GPU: {self._use_gpu}"
        )


# ============================================================================
# STANDALONE UTILITY FUNCTIONS
# ============================================================================


def check_pyvista_installation() -> Dict[str, Any]:
    """
    Check PyVista installation and capabilities.

    Returns:
        Dictionary with installation status:
            - pyvista: version and availability
            - vtk: VTK backend version
            - gpu: GPU acceleration support
            - status: Overall status message

    Examples:
        >>> status = check_pyvista_installation()
        >>> print(status["pyvista"]["version"])
        '0.46.3'
    """
    status = {
        "pyvista": {
            "available": PYVISTA_AVAILABLE,
            "version": PYVISTA_VERSION,
        },
        "vtk": {
            "version": VTK_VERSION,
        },
    }

    if PYVISTA_AVAILABLE:
        status["status"] = "PyVista ready for 3D visualization"
        status["install_command"] = "Already installed ✓"

        # Check GPU capabilities
        try:
            import vtk
            render_window = vtk.vtkRenderWindow()
            renderer = render_window.MakeRenderWindow()
            status["gpu"] = {
                "available": True,
                "renderer": renderer.GetClassName(),
            }
        except:
            status["gpu"] = {"available": False}
    else:
        status["status"] = "PyVista not installed"
        status["install_command"] = "pip install 'pymemorial[viz3d]'"

    return status


def create_example_truss() -> Tuple[NDArrayFloat, List[Tuple[int, int]]]:
    """
    Create example 3D truss structure for testing.

    Returns:
        Tuple of (nodes, members)

    Examples:
        >>> nodes, members = create_example_truss()
        >>> engine = PyVistaEngine()
        >>> mesh = engine.add_frame_members(nodes, members)
    """
    # Simple 3D truss (4 nodes, 6 members)
    nodes = np.array([
        [0, 0, 0],
        [4, 0, 0],
        [4, 3, 0],
        [0, 3, 0],
        [2, 1.5, 3],  # Apex
    ])

    members = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Base
        (0, 4), (1, 4), (2, 4), (3, 4),  # Legs
    ]

    return nodes, members


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main class
    "PyVistaEngine",
    # Utilities
    "check_pyvista_installation",
    "create_example_truss",
    # Constants
    "PYVISTA_AVAILABLE",
    "PYVISTA_VERSION",
    "VTK_VERSION",
    "STRESS_COLORMAPS",
    "MATERIAL_RENDER_STYLES",
    "CAMERA_PRESETS",
]
