# %% [markdown]
# # Basis Optimization Demo
#
# This notebook demonstrates the current MASCAF basis-optimization workflow.
#
# A downsampled morphology basis can be refined with `BasisOptimizer`, or
# directly through `CableFitter` via `FitOptions(basis_optimizer_options=...)`.

# %%
from mascaf import (
    BasisOptimizer,
    BasisOptimizerOptions,
    CableFitter,
    FitOptions,
    MeshManager,
    MorphologyGraph,
    SkeletonGraph,
    example_mesh,
)
import numpy as np

# %% [markdown]
# ## Example 1: Inspect a basis optimization directly

# %%
mesh = example_mesh("cylinder", radius=1.0, height=10.0, sections=32)
mesh_manager = MeshManager(mesh)

skeleton_points = np.array(
    [
        [0.4, 0.3, -4.0],
        [0.3, 0.2, -2.0],
        [0.2, 0.1, 0.0],
        [0.3, 0.2, 2.0],
        [0.4, 0.3, 4.0],
    ]
)
skeleton = SkeletonGraph.from_polylines([skeleton_points])

fit_options = FitOptions(max_edge_length=2.0, radius_strategy="equivalent_area")
initial_basis = MorphologyGraph.from_skeleton_graph_resample(
    skeleton,
    fit_options.max_edge_length,
)

optimizer_options = BasisOptimizerOptions(
    do_pruning=False,
    do_snapping=True,
    do_forcing=True,
    max_iterations=50,
    lambda_centering=0.5,
    lambda_smoothing=0.5,
    preserve_terminal_nodes=True,
)

optimizer = BasisOptimizer(initial_basis, mesh, optimizer_options)
optimized_basis = optimizer.optimize()
stats = optimizer.get_optimization_stats()

print("Basis optimization statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

mesh_manager.visualize_mesh_3d(skel=skeleton)

# %% [markdown]
# ## Example 2: Run the full cable fitting pipeline

# %%
pipeline_options = FitOptions(
    max_edge_length=2.0,
    radius_strategy="equivalent_area",
    basis_optimizer_options=optimizer_options,
)

morphology = CableFitter(pipeline_options).fit(mesh, skeleton)
print(
    f"Morphology graph: {morphology.number_of_nodes()} nodes, "
    f"{morphology.number_of_edges()} edges"
)

# %% [markdown]
# ## Example 3: Compare optimizer settings

# %%
lambda_smoothings = [0.0, 0.3, 0.5, 0.7, 0.9]
results = []

for lambda_smoothing in lambda_smoothings:
    test_options = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=True,
        do_forcing=True,
        max_iterations=50,
        lambda_centering=0.5,
        lambda_smoothing=lambda_smoothing,
        preserve_terminal_nodes=True,
    )
    test_optimizer = BasisOptimizer(initial_basis, mesh, test_options)
    test_optimizer.optimize()
    results.append(
        {
            "lambda_smoothing": lambda_smoothing,
            "stats": test_optimizer.get_optimization_stats(),
        }
    )

print("Effect of lambda_smoothing on basis optimization:")
print("\nlambda_smoothing | Nodes Outside Mesh | Total Length")
print("-" * 55)
for result in results:
    print(
        f"      {result['lambda_smoothing']:.1f}        | "
        f"        {result['stats']['nodes_outside_mesh']:>2}         | "
        f"   {result['stats']['total_length']:.4f}"
    )

# %%
print("\nDemo complete! The current workflow now:")
print("  ✓ Refines a morphology basis with BasisOptimizer")
print("  ✓ Integrates basis refinement into CableFitter")
print("  ✓ Computes radii after basis optimization")
