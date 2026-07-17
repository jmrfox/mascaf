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
DEFAULT_OUTPUT_DIR = Path("outputs")


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
        default=None,
        help=(
            "CableFitter max edge length. Mutually exclusive with "
            "--max-edge-length-fraction."
        ),
    )
    parser.add_argument(
        "--max-edge-length-fraction",
        type=float,
        default=None,
        help=(
            "Set max edge length as this fraction of the mesh bounding "
            "box diagonal. Mutually exclusive with --max-edge-length."
        ),
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
    parser.add_argument(
        "--log-validation",
        action="store_true",
        help=(
            "Write full_validation() output to validation.log in output-dir."
        ),  # noqa: E501
    )
    return parser.parse_args()


def resolve_max_edge_length(
    args: argparse.Namespace,
    mesh_manager: MeshManager,
) -> float:
    both_set = (
        args.max_edge_length is not None and args.max_edge_length_fraction is not None
    )
    if both_set:
        raise ValueError(
            "--max-edge-length and --max-edge-length-fraction are mutually "
            "exclusive."
        )
    if args.max_edge_length_fraction is not None:
        diagonal = mesh_manager.bounding_box_diagonal()
        return args.max_edge_length_fraction * diagonal
    if args.max_edge_length is not None:
        return args.max_edge_length
    diagonal = mesh_manager.bounding_box_diagonal()
    return 0.1 * diagonal


def build_fit_options(args: argparse.Namespace) -> FitOptions:
    basis_options = None
    if not args.disable_basis_optimization:
        basis_options = BasisOptimizerOptions(
            do_snapping=True,
            do_forcing=False,
            max_iterations=10,
            alpha_s=0.1,
        )
    return FitOptions(
        max_edge_length=args._resolved_max_edge_length,
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
    args._resolved_max_edge_length = resolve_max_edge_length(args, mesh_manager)
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

    if args.log_validation:
        log_path = output_dir / "validation.log"
        mascaf_logger = logging.getLogger("mascaf")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        mascaf_logger.addHandler(file_handler)
        mascaf_logger.setLevel(logging.INFO)
        validator.full_validation()
        mascaf_logger.removeHandler(file_handler)
        file_handler.close()

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
