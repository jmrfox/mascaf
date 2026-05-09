from __future__ import annotations

import argparse
import logging
from importlib.resources import files
from pathlib import Path

from mascaf import (
    BasisOptimizerOptions,
    CGALConfig,
    CGALError,
    CGALExecutableNotFoundError,
    CGALOperator,
    CableFitter,
    FitOptions,
    MeshManager,
    SkeletonGraph,
    Validation,
)

_DEMO = files("mascaf.demo")
DEFAULT_MESH = Path(str(_DEMO / "torus.obj"))
DEFAULT_FALLBACK_SKELETON = Path(str(_DEMO / "torus.polylines.txt"))
DEFAULT_OUTPUT_DIR = Path.home() / ".mascaf" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MASCAF pipeline on a mesh, with optional " "CGAL skeletonization."
        ),
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(DEFAULT_MESH),
        help=("Path to the input mesh. Defaults to " "data/demo/torus.obj."),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=("Directory for generated outputs."),
    )
    parser.add_argument(
        "--skeleton-path",
        default=None,
        help=("Use an existing skeleton file instead of generating one."),
    )
    parser.add_argument(
        "--skip-cgal",
        action="store_true",
        help=("Do not attempt internal CGAL skeletonization."),
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=0.5,
        help=("CableFitter max edge length."),
    )
    parser.add_argument(
        "--radius-strategy",
        default="equivalent_area",
        help=("Radius strategy for FitOptions."),
    )
    parser.add_argument(
        "--quality-speed-tradeoff",
        type=float,
        default=0.5,
        help=("CGAL skeletonization quality_speed_tradeoff (w_H)."),
    )
    parser.add_argument(
        "--medially-centered-speed-tradeoff",
        type=float,
        default=5.0,
        help=("CGAL skeletonization medially_centered_speed_tradeoff (w_M)."),
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help=("Optional CGAL build directory to help locate executables."),
    )
    parser.add_argument(
        "--executable-dir",
        default=None,
        help=("Optional directory containing built CGAL executables."),
    )
    parser.add_argument(
        "--disable-basis-optimization",
        action="store_true",
        help=("Skip BasisOptimizer during fitting."),
    )
    parser.add_argument(
        "--scale-metric",
        choices=["surface_area", "volume"],
        default=None,
        help=("Optionally scale radii to match the mesh after fitting."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Enable INFO-level logging."),
    )
    return parser.parse_args()


def build_fit_options(args: argparse.Namespace) -> FitOptions:
    basis_options = None
    if not args.disable_basis_optimization:
        basis_options = BasisOptimizerOptions(
            do_snapping=True,
            do_forcing=True,
            max_iterations=50,
            smoothing_weight=0.5,
        )
    return FitOptions(
        max_edge_length=args.max_edge_length,
        radius_strategy=args.radius_strategy,
        basis_optimizer_options=basis_options,
    )


def resolve_skeleton_path(
    args: argparse.Namespace,
    mesh_path: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    if args.skeleton_path:
        skeleton_path = Path(args.skeleton_path).expanduser().resolve()
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_path}")
        return skeleton_path, "user-supplied skeleton"

    if not args.skip_cgal:
        config = CGALConfig.from_overrides(
            build_dir=args.build_dir,
            executable_dir=args.executable_dir,
        )
        operator = CGALOperator(config=config)
        generated_skeleton = output_dir / f"{mesh_path.stem}.polylines.txt"
        try:
            operator.skeletonize(
                mesh_path,
                generated_skeleton,
                quality_speed_tradeoff=args.quality_speed_tradeoff,
                medially_centered_speed_tradeoff=(
                    args.medially_centered_speed_tradeoff
                ),
            )
            return generated_skeleton, "internal CGAL skeletonization"
        except (CGALExecutableNotFoundError, CGALError) as exc:
            logging.warning(
                "Internal CGAL skeletonization unavailable: %s",
                exc,
            )

    if (
        mesh_path.resolve() == DEFAULT_MESH.resolve()
        and DEFAULT_FALLBACK_SKELETON.exists()
    ):
        return DEFAULT_FALLBACK_SKELETON, "bundled demo skeleton"

    raise RuntimeError(
        "No skeleton source is available. Provide --skeleton-path "
        "or build the internal CGAL executables."
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path, skeleton_source = resolve_skeleton_path(
        args,
        mesh_path,
        output_dir,
    )

    mesh_manager = MeshManager(mesh_path=str(mesh_path))
    skeleton = SkeletonGraph.from_txt(str(skeleton_path))
    fit_options = build_fit_options(args)
    morphology = CableFitter(fit_options).fit(mesh_manager, skeleton)

    if args.scale_metric is not None:
        morphology.scale_radii_to_match_mesh(
            mesh_manager,
            metric=args.scale_metric,
        )

    swc_path = output_dir / f"{mesh_path.stem}.swc"
    morphology.to_swc_file(str(swc_path))

    validator = Validation(mesh_manager, skeleton, morphology)
    volume_result = validator.compare_volumes()
    area_result = validator.compare_surface_areas()

    print(f"Mesh: {mesh_path}")
    print(f"Skeleton: {skeleton_path} ({skeleton_source})")
    print(f"SWC: {swc_path}")
    print(
        "MorphologyGraph: "
        f"{morphology.number_of_nodes()} nodes, "
        f"{morphology.number_of_edges()} edges"
    )
    print(
        "Volume ratio: "
        f"{volume_result['ratio']:.4f} "
        f"(relative error {volume_result['relative_error']:.2%})"
    )
    print(
        "Surface area ratio: "
        f"{area_result['ratio']:.4f} "
        f"(relative error {area_result['relative_error']:.2%})"
    )


if __name__ == "__main__":
    main()
