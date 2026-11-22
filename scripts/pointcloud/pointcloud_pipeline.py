from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


SEED = 20260901
TOLERANCE_MM = 0.50
HETEROGENEITY_VOXEL_MM = 2.0
MIN_CLUSTER_VOXELS = 2
MAX_ANALYSIS_POINTS = 120_000
REFERENCE_POINTS = 120_000
CANDIDATES = [(0.3, 4.0), (0.5, 3.0), (0.8, 3.0), (1.0, 2.5), (1.0, 4.0)]


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        o3d.utility.random.seed(seed)
    except Exception:
        pass


def cap_cloud(pcd: o3d.geometry.PointCloud, limit: int, seed: int) -> o3d.geometry.PointCloud:
    if len(pcd.points) <= limit:
        return pcd
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pcd.points), limit, replace=False))
    return pcd.select_by_index(idx.tolist())


def load_sample(path: Path) -> tuple[o3d.geometry.PointCloud, dict]:
    raw = o3d.io.read_point_cloud(str(path))
    points = np.asarray(raw.points)
    if len(points) < 100:
        raise ValueError(f"Too few points in {path}")
    finite = np.all(np.isfinite(points), axis=1)
    raw = raw.select_by_index(np.flatnonzero(finite).tolist())
    before_cap = len(raw.points)
    raw = cap_cloud(raw, 300_000, SEED + 1)
    before_sor = len(raw.points)
    _, inliers = raw.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
    clean = raw.select_by_index(inliers)
    clean = cap_cloud(clean, MAX_ANALYSIS_POINTS, SEED + 2)
    qc = {
        "raw_points": int(len(points)),
        "finite_points": int(np.count_nonzero(finite)),
        "pre_sor_points": int(before_sor),
        "analysis_points": int(len(clean.points)),
        "initial_cap_removed": int(max(0, before_cap - before_sor)),
        "sor_removed": int(before_sor - len(inliers)),
    }
    return clean, qc


def load_reference(path: Path) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, dict]:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(loaded.vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(loaded.faces, dtype=np.int32))
    if len(mesh.vertices) < 4 or len(mesh.triangles) < 4:
        raise ValueError(f"Invalid STL mesh: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    set_seed(SEED)
    reference = mesh.sample_points_uniformly(number_of_points=REFERENCE_POINTS)
    reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=60))
    vertices = np.asarray(mesh.vertices)
    qc = {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "watertight": bool(mesh.is_watertight()),
        "aabb_extent_mm": np.ptp(vertices, axis=0).tolist(),
        "sampled_reference_points": int(len(reference.points)),
    }
    return mesh, reference, qc


def ensure_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(radius, 0.3), max_nn=60)
    )
    pcd.normalize_normals()


def fpfh(pcd: o3d.geometry.PointCloud, voxel: float):
    ensure_normals(pcd, voxel * 3.0)
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )


def registration_candidate(
    sample: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    voxel: float,
    multiplier: float,
    seed: int,
) -> dict:
    set_seed(seed)
    s_down = sample.voxel_down_sample(voxel)
    r_down = reference.voxel_down_sample(voxel)
    if min(len(s_down.points), len(r_down.points)) < 20:
        raise RuntimeError("Too few downsampled points")
    s_feat, r_feat = fpfh(s_down, voxel), fpfh(r_down, voxel)
    threshold = voxel * multiplier
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s_down,
        r_down,
        s_feat,
        r_feat,
        True,
        threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.90),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(60_000, 0.999),
    )
    transform = coarse.transformation.copy()
    for scale in (max(voxel * 1.5, 0.25), max(voxel * 0.8, 0.15), max(voxel * 0.4, 0.08)):
        s = sample.voxel_down_sample(scale)
        r = reference.voxel_down_sample(scale)
        ensure_normals(s, scale * 3.0)
        ensure_normals(r, scale * 3.0)
        fine = o3d.pipelines.registration.registration_icp(
            s,
            r,
            max_correspondence_distance=scale * 2.5,
            init=transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=120
            ),
        )
        transform = fine.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(
        sample, reference, TOLERANCE_MM, transform
    )
    return {
        "voxel_mm": voxel,
        "distance_multiplier": multiplier,
        "seed": seed,
        "fitness": float(evaluation.fitness),
        "inlier_rmse_mm": float(evaluation.inlier_rmse),
        "coarse_fitness": float(coarse.fitness),
        "coarse_rmse_mm": float(coarse.inlier_rmse),
        "transform": transform,
    }


def pca_initializations(sample: o3d.geometry.PointCloud, reference: o3d.geometry.PointCloud):
    source_obb = sample.get_oriented_bounding_box(robust=True)
    target_obb = reference.get_oriented_bounding_box(robust=True)
    source_r = np.asarray(source_obb.R)
    target_r = np.asarray(target_obb.R)
    source_c = np.asarray(source_obb.center)
    target_c = np.asarray(target_obb.center)
    transforms = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3))
        for target_axis, source_axis in enumerate(perm):
            base[target_axis, source_axis] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = np.diag(signs) @ base
            rotation = target_r @ mapping @ source_r.T
            if np.linalg.det(rotation) < 0.0:
                continue
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_c - rotation @ source_c
            transforms.append(transform)
    return transforms


def pca_registration_candidates(sample: o3d.geometry.PointCloud, reference: o3d.geometry.PointCloud) -> list[dict]:
    sample_quick = sample.voxel_down_sample(1.0)
    reference_quick = reference.voxel_down_sample(1.0)
    quick = []
    for index, init in enumerate(pca_initializations(sample_quick, reference_quick), 1):
        result = o3d.pipelines.registration.registration_icp(
            sample_quick,
            reference_quick,
            3.0,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        evaluation = o3d.pipelines.registration.evaluate_registration(
            sample_quick, reference_quick, 1.0, result.transformation
        )
        quick.append((float(evaluation.fitness), float(evaluation.inlier_rmse), result.transformation, index))
    quick.sort(key=lambda x: (-x[0], x[1]))
    refined = []
    for rank, (_, _, init, index) in enumerate(quick[:4], 1):
        transform = init.copy()
        for scale, distance in ((1.0, 2.5), (0.6, 1.5), (0.3, 0.75), (0.18, 0.50)):
            s = sample.voxel_down_sample(scale)
            r = reference.voxel_down_sample(scale)
            ensure_normals(s, scale * 3.0)
            ensure_normals(r, scale * 3.0)
            result = o3d.pipelines.registration.registration_icp(
                s,
                r,
                distance,
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=120
                ),
            )
            transform = result.transformation
        evaluation = o3d.pipelines.registration.evaluate_registration(
            sample, reference, TOLERANCE_MM, transform
        )
        refined.append(
            {
                "voxel_mm": 1.0,
                "distance_multiplier": 3.0,
                "seed": SEED,
                "fitness": float(evaluation.fitness),
                "inlier_rmse_mm": float(evaluation.inlier_rmse),
                "coarse_fitness": float(quick[rank - 1][0]),
                "coarse_rmse_mm": float(quick[rank - 1][1]),
                "transform": transform,
                "initialization": f"obb_pca_candidate_{index}",
            }
        )
    return refined


def register(sample: o3d.geometry.PointCloud, reference: o3d.geometry.PointCloud) -> tuple[dict, list[dict]]:
    log("registration OBB/PCA orientation candidates")
    candidates = pca_registration_candidates(sample, reference)
    candidates.sort(key=lambda x: (-x["fitness"], x["inlier_rmse_mm"]))
    if not candidates or candidates[0]["fitness"] < 0.50:
        for i, (voxel, multiplier) in enumerate(CANDIDATES, 1):
            log(f"registration FPFH candidate {i}/{len(CANDIDATES)} voxel={voxel} multiplier={multiplier}")
            try:
                candidate = registration_candidate(sample, reference, voxel, multiplier, SEED + i * 17)
                candidate["initialization"] = "fpfh_ransac"
                candidates.append(candidate)
            except Exception as exc:
                log(f"candidate {i} failed: {exc}")
    if not candidates:
        raise RuntimeError("All registration candidates failed")
    candidates.sort(key=lambda x: (-x["fitness"], x["inlier_rmse_mm"]))
    return candidates[0], candidates


def signed_distance(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> tuple[np.ndarray, str]:
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    chunks = []
    method = "watertight_stl_raycasting_signed_distance"
    for start in range(0, len(points), 100_000):
        query = o3d.core.Tensor(points[start : start + 100_000].astype(np.float32))
        if mesh.is_watertight():
            chunks.append(scene.compute_signed_distance(query).numpy().astype(float))
        else:
            closest = scene.compute_closest_points(query)
            q = closest["points"].numpy().astype(float)
            n = closest["primitive_normals"].numpy().astype(float)
            delta = points[start : start + 100_000] - q
            distance = np.linalg.norm(delta, axis=1)
            sign = np.where(np.einsum("ij,ij->i", delta, n) >= 0, 1.0, -1.0)
            chunks.append(sign * distance)
            method = "triangle_normal_signed_distance_nonwatertight_fallback"
    return np.concatenate(chunks), method


def reference_frame_dimensions(mesh: o3d.geometry.TriangleMesh, sample_points: np.ndarray) -> dict:
    vertices = np.asarray(mesh.vertices)
    obb = mesh.get_oriented_bounding_box(robust=True)
    center, rotation = np.asarray(obb.center), np.asarray(obb.R)
    ref_local = (vertices - center) @ rotation
    sample_local = (sample_points - center) @ rotation
    reference_extent = np.ptp(ref_local, axis=0)
    low, high = np.percentile(sample_local, [0.25, 99.75], axis=0)
    sample_extent = high - low
    order = np.argsort(reference_extent)[::-1]
    ref_lwh = reference_extent[order]
    sample_lwh = sample_extent[order]
    err = sample_lwh - ref_lwh
    err_pct = err / np.maximum(ref_lwh, 1e-12) * 100.0
    return {
        "reference_lwh_mm": ref_lwh.tolist(),
        "sample_lwh_mm": sample_lwh.tolist(),
        "error_lwh_mm": err.tolist(),
        "error_lwh_percent": err_pct.tolist(),
        "mean_absolute_dimension_error_percent": float(np.mean(np.abs(err_pct))),
        "global_dimension_fidelity_percent": float(max(0.0, 100.0 - np.mean(np.abs(err_pct)))),
        "sample_percentile_range": [0.25, 99.75],
    }


def macro_coverage(reference_points: np.ndarray, reverse_distance: np.ndarray) -> dict:
    voxel = max(HETEROGENEITY_VOXEL_MM, TOLERANCE_MM)
    origin = reference_points.min(axis=0)
    idx = np.floor((reference_points - origin) / voxel).astype(np.int64)
    _, inverse = np.unique(idx, axis=0, return_inverse=True)
    total = np.bincount(inverse)
    covered = np.bincount(inverse, weights=(reverse_distance <= TOLERANCE_MM).astype(float))
    local = covered / np.maximum(total, 1)
    return {
        "macro_coverage": float(np.mean(local >= 0.5)),
        "voxel_mm": voxel,
        "reference_voxels": int(len(local)),
    }


def spatial_metrics(points: np.ndarray, signed: np.ndarray) -> dict:
    voxel = HETEROGENEITY_VOXEL_MM
    origin = points.min(axis=0)
    idx = np.floor((points - origin) / voxel).astype(np.int64)
    unique, inverse = np.unique(idx, axis=0, return_inverse=True)
    absolute = np.abs(signed)
    sums = np.bincount(inverse, weights=absolute)
    counts = np.bincount(inverse)
    local_mean = sums / np.maximum(counts, 1)
    mean_local = float(np.mean(local_mean)) if len(local_mean) else 0.0
    hotspot = local_mean > TOLERANCE_MM
    hotspot_coords = {tuple(map(int, x)) for x in unique[hotspot]}
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    sizes = []
    unseen = set(hotspot_coords)
    while unseen:
        seed = unseen.pop()
        stack, size = [seed], 1
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        sizes.append(size)
    valid = [x for x in sizes if x >= MIN_CLUSTER_VOXELS]
    largest = max(valid) if valid else 0
    return {
        "voxel_mm": voxel,
        "occupied_voxels": int(len(unique)),
        "heterogeneity_cv": float(np.std(local_mean) / (mean_local + 1e-12)),
        "hotspot_voxel_ratio": float(np.mean(hotspot)) if len(hotspot) else 0.0,
        "cluster_count": int(len(valid)),
        "largest_cluster_voxels": int(largest),
        "largest_cluster_support_mm3": float(largest * voxel**3),
        "connectivity": 26,
        "minimum_cluster_voxels": MIN_CLUSTER_VOXELS,
    }


def analyse(mesh, reference, registered, best, sample_qc, reference_qc) -> tuple[dict, np.ndarray]:
    sample_points = np.asarray(registered.points)
    reference_points = np.asarray(reference.points)
    signed, signed_method = signed_distance(mesh, sample_points)
    forward = np.abs(signed)
    reverse, _ = cKDTree(sample_points).query(reference_points, k=1, workers=-1)
    precision = float(np.mean(forward <= TOLERANCE_MM))
    recall = float(np.mean(reverse <= TOLERANCE_MM))
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    absolute = np.abs(signed)
    result = {
        "run": {
            "seed": SEED,
            "tolerance_mm": TOLERANCE_MM,
            "heterogeneity_voxel_mm": HETEROGENEITY_VOXEL_MM,
            "max_analysis_points": MAX_ANALYSIS_POINTS,
        },
        "input_qc": {"sample": sample_qc, "reference": reference_qc},
        "registration": {k: v for k, v in best.items() if k != "transform"},
        "d1_global_geometric_fidelity": reference_frame_dimensions(mesh, sample_points),
        "d2_structural_completeness": {
            "sample_precision": precision,
            "cad_recall": recall,
            "f_score": fscore,
            **macro_coverage(reference_points, reverse),
        },
        "d3_surface_deviation": {
            "mae_mm": float(np.mean(absolute)),
            "rmse_mm": float(np.sqrt(np.mean(signed**2))),
            "p95_abs_mm": float(np.percentile(absolute, 95)),
            "max_abs_mm": float(np.max(absolute)),
            "symmetric_chamfer_mean_mm": float(0.5 * (np.mean(forward) + np.mean(reverse))),
            "symmetric_hausdorff95_mm": float(max(np.percentile(forward, 95), np.percentile(reverse, 95))),
        },
        "d4_deviation_polarity_tolerance": {
            "signed_distance_method": signed_method,
            "mean_signed_mm": float(np.mean(signed)),
            "median_signed_mm": float(np.median(signed)),
            "positive_ratio": float(np.mean(signed > 0)),
            "negative_ratio": float(np.mean(signed < 0)),
            "within_tolerance_ratio": precision,
            "beyond_tolerance_ratio": 1.0 - precision,
        },
        "d5_spatial_heterogeneity": spatial_metrics(sample_points, signed),
    }
    return result, signed


def json_safe_candidate(item: dict) -> dict:
    return {**{k: v for k, v in item.items() if k != "transform"}, "transform": item["transform"].tolist()}


def write_flat_metrics(path: Path, result: dict) -> None:
    rows = []
    for domain, values in result.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)):
                rows.append((domain, key, value))
            elif isinstance(value, list):
                rows.append((domain, key, json.dumps(value, ensure_ascii=False)))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "metric", "value"])
        writer.writerows(rows)


def run_case(sample_path: Path, stl_path: Path, output_dir: Path, case_id: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading case={case_id}")
    sample, sample_qc = load_sample(sample_path)
    mesh, reference, reference_qc = load_reference(stl_path)
    log(f"sample points={len(sample.points):,}; reference points={len(reference.points):,}")
    best, candidates = register(sample, reference)
    log(f"best fitness={best['fitness']:.6f}; RMSE={best['inlier_rmse_mm']:.6f} mm")
    registered = o3d.geometry.PointCloud(sample)
    registered.transform(best["transform"])
    result, signed = analyse(mesh, reference, registered, best, sample_qc, reference_qc)
    result["case_id"] = case_id
    result["inputs"] = {
        "sample": str(sample_path),
        "sample_sha256": sha256(sample_path),
        "reference": str(stl_path),
        "reference_sha256": sha256(stl_path),
    }
    result["registration_transform"] = best["transform"].tolist()
    result["registration_candidates"] = [json_safe_candidate(x) for x in candidates]
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_flat_metrics(output_dir / "metrics_long.csv", result)
    np.savez_compressed(
        output_dir / "pointwise_deviation.npz",
        points=np.asarray(registered.points).astype(np.float32),
        signed_deviation_mm=signed.astype(np.float32),
    )
    o3d.io.write_point_cloud(str(output_dir / "sample_cleaned.ply"), sample, write_ascii=False)
    o3d.io.write_point_cloud(str(output_dir / "sample_registered.ply"), registered, write_ascii=False)
    o3d.io.write_point_cloud(str(output_dir / "reference_sampled.ply"), reference, write_ascii=False)
    log(f"completed case={case_id}; outputs={output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    run_case(args.sample, args.stl, args.output, args.case_id)


if __name__ == "__main__":
    main()
