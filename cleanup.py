from pathlib import Path
from itertools import combinations, permutations
import numpy as np
import meshio as mio
import igl
import vtk
from vtk.util.numpy_support import vtk_to_numpy

BASE = Path("/Users/teseo/data/zebrafish/zebra-good")
RUNS = BASE / "runs"
OBJ = BASE / "high_quality_manifold.obj"

boundary = mio.read(OBJ)
Vb = boundary.points
Fb = boundary.cells_dict["triangle"]

EPS = 0.05         # tune: 0.02 stricter, 0.1 looser
FACE_CHUNK = 500_000  # faces per winding-number batch
POINT_DATA_ARRAYS = None  # all point data: None; geometry only: []; selected: ["u"]
TET_WN_THRESHOLD = 0.0
SURFACE_WN_TARGET = 0.5
BOUNDARY_FACE_EPS = 0.5
FILTER_BOUNDARY_BY_WN = False
STITCH_DECIMALS = 12
WRITE_SELECTED_TETS = False
OVERWRITE_OUTPUTS = True


def tet_faces(T, cell_type):
    """Return linear faces for tests and possibly quadratic faces for output."""
    if cell_type == "tetra":
        F = np.vstack([
            T[:, [0, 2, 1]],
            T[:, [0, 1, 3]],
            T[:, [1, 2, 3]],
            T[:, [2, 0, 3]],
        ])
        return F, F, "triangle"

    if cell_type == "tetra10":
        # meshio/VTK tetra10 node order:
        # 0,1,2,3 corners; 4=(0,1), 5=(1,2), 6=(2,0),
        # 7=(0,3), 8=(1,3), 9=(2,3).
        F_quad = np.vstack([
            T[:, [0, 2, 1, 6, 5, 4]],
            T[:, [0, 1, 3, 4, 8, 7]],
            T[:, [1, 2, 3, 5, 9, 8]],
            T[:, [2, 0, 3, 6, 7, 9]],
        ])
        return F_quad[:, :3], F_quad, "triangle6"

    raise ValueError(f"unsupported tet cell type: {cell_type}")


def _best_tetra10_order_from_sample(V, T_sample):
    edge_pairs = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 3],
        [2, 3],
    ])

    X = V[T_sample]
    best_order = None
    best_score = np.inf

    for corner_positions in combinations(range(10), 4):
        remaining_positions = [p for p in range(10) if p not in corner_positions]

        for ordered_corners in permutations(corner_positions):
            corners = X[:, ordered_corners, :]
            candidates = X[:, remaining_positions, :]

            midpoints = 0.5 * (
                corners[:, edge_pairs[:, 0], :] + corners[:, edge_pairs[:, 1], :]
            )
            diff = midpoints[:, :, None, :] - candidates[:, None, :, :]
            mean_dist2 = np.mean(np.sum(diff * diff, axis=3), axis=0)
            edge_to_remaining = np.argmin(mean_dist2, axis=1)

            if len(np.unique(edge_to_remaining)) != 6:
                continue

            score = mean_dist2[np.arange(6), edge_to_remaining].mean()
            if score < best_score:
                best_score = score
                best_order = np.array(
                    list(ordered_corners)
                    + [remaining_positions[j] for j in edge_to_remaining],
                    dtype=np.int64,
                )

    if best_order is None:
        raise ValueError("could not infer tetra10 node order from coordinates")

    return best_order, best_score


def normalize_tetra10_order(V, T):
    """Reorder 10-node tets to VTK quadratic tetra edge-node order."""
    sample = T[:min(len(T), 1000)]
    if len(sample) == 0:
        return T

    order, score = _best_tetra10_order_from_sample(V, sample)
    print(f"inferred tetra10 node order {order.tolist()} with mean midpoint error {score:g}")
    return T[:, order]


def vtk_point_data_to_numpy(grid, vtk_to_numpy):
    point_data = grid.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        if array is None:
            continue

        name = array.GetName() or f"point_data_{i}"
        if POINT_DATA_ARRAYS is not None and name not in POINT_DATA_ARRAYS:
            continue

        arrays[name] = vtk_to_numpy(array).copy()

    return arrays


def print_vtk_array_names(reader):
    point_selection = reader.GetPointDataArraySelection()
    cell_selection = reader.GetCellDataArraySelection()

    point_names = [
        point_selection.GetArrayName(i)
        for i in range(point_selection.GetNumberOfArrays())
    ]
    cell_names = [
        cell_selection.GetArrayName(i)
        for i in range(cell_selection.GetNumberOfArrays())
    ]

    print(f"available point data arrays: {point_names}")
    print(f"available cell data arrays: {cell_names}")


def read_vtu_geometry(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    print_vtk_array_names(reader)
    if POINT_DATA_ARRAYS is None:
        reader.GetPointDataArraySelection().EnableAllArrays()
    else:
        reader.GetPointDataArraySelection().DisableAllArrays()
        for name in POINT_DATA_ARRAYS:
            reader.GetPointDataArraySelection().EnableArray(name)

    reader.GetCellDataArraySelection().DisableAllArrays()
    reader.Update()

    grid = reader.GetOutput()
    V = vtk_to_numpy(grid.GetPoints().GetData())
    point_data = vtk_point_data_to_numpy(grid, vtk_to_numpy)
    print(f"loaded point data arrays: {list(point_data)}")

    vtk_cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    unique_cell_types = np.unique(vtk_cell_types)
    if len(unique_cell_types) != 1:
        raise ValueError(f"expected one cell type, got VTK cell types {unique_cell_types}")

    vtk_cell_type = int(unique_cell_types[0])
    if vtk_cell_type == vtk.VTK_TETRA:
        tet_cell_type = "tetra"
        nodes_per_cell = 4
    elif vtk_cell_type in (vtk.VTK_QUADRATIC_TETRA, vtk.VTK_LAGRANGE_TETRAHEDRON):
        tet_cell_type = "tetra10"
    else:
        raise ValueError(f"unsupported VTK cell type: {vtk_cell_type}")

    cells = grid.GetCells()
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    cell_sizes = np.diff(offsets)
    unique_cell_sizes = np.unique(cell_sizes)
    if len(unique_cell_sizes) != 1:
        raise ValueError(f"expected fixed-size cells, got sizes {unique_cell_sizes}")

    nodes_per_cell = int(unique_cell_sizes[0])
    if tet_cell_type == "tetra" and nodes_per_cell != 4:
        raise ValueError(f"expected 4-node tetra cells, got {nodes_per_cell} nodes")
    if tet_cell_type == "tetra10" and nodes_per_cell != 10:
        raise ValueError(
            f"expected 10-node quadratic/Lagrange tetra cells, got {nodes_per_cell} nodes"
        )

    T = connectivity.reshape(-1, nodes_per_cell)
    if tet_cell_type == "tetra10":
        T = normalize_tetra10_order(V, T)

    # Keep VTK-owned arrays alive by copying only the arrays needed downstream.
    return V.copy(), T.copy(), tet_cell_type, point_data


def compact_mesh(V, F, point_data):
    used, inverse = np.unique(F.reshape(-1), return_inverse=True)
    F_new = inverse.reshape(F.shape)

    V_new = V[used]

    point_data_new = {}
    for name, data in point_data.items():
        point_data_new[name] = data[used]

    return V_new, F_new, point_data_new


def write_selected_tets(path, V, T_selected, tet_cell_type):
    if len(T_selected) == 0:
        print(f"no selected tets to write: {path}")
        return

    print(f"writing selected WN tets to {path}")
    V_new, T_new, _ = compact_mesh(V, T_selected, {})
    mio.Mesh(
        points=V_new,
        cells=[(tet_cell_type, T_new)],
    ).write(path)


def classify_tets_by_barycenter_winding(V, T):
    inside_chunks = []
    tet_chunk = max(1, FACE_CHUNK // 4)

    print("filtering tets by barycenter winding number")
    for i in range(0, len(T), tet_chunk):
        T_i = T[i:i + tet_chunk]
        bary = np.mean(V[T_i[:, :4]], axis=1)
        wn = igl.fast_winding_number(Vb, Fb, bary)
        inside_chunks.append(wn > TET_WN_THRESHOLD)
        print(f"  {i} / {len(T)}")

    return np.concatenate(inside_chunks)


def stitched_face_key(V, corner_face):
    coords = np.round(V[corner_face], STITCH_DECIMALS)
    return tuple(sorted(tuple(row) for row in coords))


def boundary_faces_from_stitched_selected_tets(V, T_selected, tet_cell_type):
    face_map = {}
    tet_chunk = max(1, FACE_CHUNK // 4)

    print("finding stitched boundary faces of selected tets")
    for i in range(0, len(T_selected), tet_chunk):
        T_i = T_selected[i:i + tet_chunk]
        F_corner_i, F_out_i, _ = tet_faces(T_i, tet_cell_type)

        for corner_face, out_face in zip(F_corner_i, F_out_i):
            key = stitched_face_key(V, corner_face)
            count, face = face_map.get(key, (0, None))
            face_map[key] = (count + 1, out_face.copy())

        print(f"  {i} / {len(T_selected)}")

    boundary = [
        face for count, face in face_map.values()
        if count == 1
    ]
    if not boundary:
        return np.empty((0, 6 if tet_cell_type == "tetra10" else 3), dtype=T_selected.dtype)

    return np.asarray(boundary, dtype=T_selected.dtype)


def interface_faces_from_stitched_tets(V, T, tet_cell_type, tet_inside):
    face_map = {}
    selected_face_chunks = []
    tet_chunk = max(1, FACE_CHUNK // 4)

    print("finding stitched interface faces between selected and unselected tets")
    for i in range(0, len(T), tet_chunk):
        T_i = T[i:i + tet_chunk]
        inside_i = tet_inside[i:i + tet_chunk]
        F_corner_i, F_out_i, _ = tet_faces(T_i, tet_cell_type)
        face_inside_i = np.repeat(inside_i, 4)

        selected_i = []
        for corner_face, is_inside, out_face in zip(F_corner_i, face_inside_i, F_out_i):
            key = stitched_face_key(V, corner_face)
            previous = face_map.pop(key, None)
            if previous is None:
                face_map[key] = (bool(is_inside), out_face.copy())
                continue

            previous_inside, previous_face = previous
            if previous_inside != bool(is_inside):
                selected_i.append(out_face.copy() if is_inside else previous_face)

        if selected_i:
            selected_face_chunks.append(np.asarray(selected_i, dtype=T.dtype))

        print(f"  {i} / {len(T)}")

    if selected_face_chunks:
        return np.vstack(selected_face_chunks)

    return np.empty((0, 6 if tet_cell_type == "tetra10" else 3), dtype=T.dtype)


def filter_boundary_faces_by_winding(V, F):
    if not FILTER_BOUNDARY_BY_WN or len(F) == 0:
        return F

    keep_chunks = []

    print("filtering stitched boundary faces by face barycenter winding number")
    for i in range(0, len(F), FACE_CHUNK):
        F_i = F[i:i + FACE_CHUNK]
        bary = igl.barycenter(V, F_i[:, :3])
        wn = igl.fast_winding_number(Vb, Fb, bary)
        keep_chunks.append(np.abs(wn - SURFACE_WN_TARGET) < BOUNDARY_FACE_EPS)
        print(f"  {i} / {len(F)}")

    keep = np.concatenate(keep_chunks)
    print(f"kept {np.count_nonzero(keep)} / {len(F)} boundary faces after WN filter")
    return F[keep]


# for run_dir in sorted(RUNS.glob("index_*")):
for i in range(1):
    sim_path = Path("/Users/teseo/Downloads/sim.vtu")
    # sim_path = run_dir / "outnz" / "sim.vtu"
    out_path = Path("/Users/teseo/Downloads/touching_surface.vtu")
    selected_tets_path = Path("/Users/teseo/Downloads/wn_selected_tets.vtu")
    # out_path = run_dir / "outnz" / "touching_surface.vtu"

    if not sim_path.exists():
        print(f"missing: {sim_path}")
        continue

    if out_path.exists() and not OVERWRITE_OUTPUTS:
        print(f"skip existing: {out_path}")
        continue

    print(f"reading {sim_path}")
    V, T, tet_cell_type, point_data = read_vtu_geometry(sim_path)
    print("Done reading")

    _, _, face_cell_type = tet_faces(T[:0], tet_cell_type)

    tet_inside = classify_tets_by_barycenter_winding(V, T)
    print(f"selected tets: {np.count_nonzero(tet_inside)} / {len(tet_inside)}")

    T_selected = T[tet_inside]
    if WRITE_SELECTED_TETS:
        write_selected_tets(selected_tets_path, V, T_selected, tet_cell_type)

    F_keep = boundary_faces_from_stitched_selected_tets(V, T_selected, tet_cell_type)
    F_keep = filter_boundary_faces_by_winding(V, F_keep)

    if len(F_keep) == 0:
        print(f"no faces selected for {sim_path}")
        continue

    print(f"selected {len(F_keep)} faces; compacting vertices")
    V_new, F_new, point_data_new = compact_mesh(
        V,
        F_keep,
        point_data,
    )

    surf = mio.Mesh(
        points=V_new,
        cells=[(face_cell_type, F_new)],
        point_data=point_data_new,
    )

    print(f"writing {out_path}")
    surf.write(out_path)

    # Help memory before next huge file
    del V, T, T_selected, tet_inside, point_data, F_keep, V_new, F_new, point_data_new, surf
