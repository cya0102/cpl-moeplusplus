import argparse
import json
import os
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(raw_path, base_dir):
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(base_dir, raw_path)


def resolve_feature_path(raw_feature_path):
    """
    兼容两类路径:
    1) 配置中的绝对路径可直接使用
    2) 绝对路径失效时, 尝试映射到当前仓库下的 data 目录
    """
    direct = resolve_path(raw_feature_path, PROJECT_ROOT)
    if os.path.exists(direct):
        return direct

    marker = f"{os.sep}data{os.sep}"
    if marker in raw_feature_path:
        suffix = raw_feature_path.split(marker, 1)[1]
        fallback = os.path.join(PROJECT_ROOT, "data", suffix)
        if os.path.exists(fallback):
            return fallback

    return direct


def parse_datasets_arg(raw):
    val = raw.strip().lower()
    if val in {"activitynet", "anet"}:
        return ["activitynet"]
    if val in {"charades", "charadessta", "charades-sta"}:
        return ["charades"]
    if val in {"both", "all"}:
        return ["activitynet", "charades"]

    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    mapped = []
    for p in parts:
        if p in {"activitynet", "anet"}:
            mapped.append("activitynet")
        elif p in {"charades", "charadessta", "charades-sta"}:
            mapped.append("charades")
        else:
            raise ValueError(
                f"Unsupported dataset value: {p}. "
                "Use activitynet, charades, both, or comma-separated list."
            )
    if not mapped:
        raise ValueError("--datasets is empty after parsing.")
    return sorted(set(mapped))


def pairwise_cosine_stats(frame_features):
    """
    frame_features: [T, D]
    返回每个视频内部帧特征两两余弦相似度统计。
    """
    feats = np.asarray(frame_features, dtype=np.float32)
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)

    num_frames = feats.shape[0]
    if num_frames < 2:
        return {
            "num_frames": int(num_frames),
            "num_pairs": 0,
            "avg_similarity": None,
            "max_similarity": None,
            "min_similarity": None,
        }

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = feats / norms
    sim_matrix = normalized @ normalized.T

    tri_i, tri_j = np.triu_indices(num_frames, k=1)
    sims = sim_matrix[tri_i, tri_j]

    return {
        "num_frames": int(num_frames),
        "num_pairs": int(sims.shape[0]),
        "avg_similarity": float(np.mean(sims)),
        "max_similarity": float(np.max(sims)),
        "min_similarity": float(np.min(sims)),
    }


def read_video_features(h5_file, dataset_tag, video_id):
    if dataset_tag == "activitynet":
        return np.asarray(h5_file[video_id]["c3d_features"]).astype(np.float32)
    return np.asarray(h5_file[video_id]).astype(np.float32)


def analyze_one_dataset(dataset_tag):
    config_rel = {
        "activitynet": "config/activitynet/main.json",
        "charades": "config/charades/main.json",
    }[dataset_tag]

    config_path = os.path.join(PROJECT_ROOT, config_rel)
    config = load_json(config_path)
    raw_feature_path = config["dataset"]["feature_path"]
    feature_path = resolve_feature_path(raw_feature_path)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            f"(raw path from config: {raw_feature_path})"
        )

    samples = []
    print(f"\n[{dataset_tag}] Loading features from: {feature_path}")
    with h5py.File(feature_path, "r") as fr:
        video_ids = list(fr.keys())
        for vid in tqdm(video_ids, desc=f"Analyzing {dataset_tag}", ncols=100):
            try:
                frames_feat = read_video_features(fr, dataset_tag, vid)
                stats = pairwise_cosine_stats(frames_feat)
                samples.append(
                    {
                        "video_id": str(vid),
                        "avg_similarity": stats["avg_similarity"],
                        "max_similarity": stats["max_similarity"],
                        "min_similarity": stats["min_similarity"],
                        "num_frames": stats["num_frames"],
                        "num_pairs": stats["num_pairs"],
                    }
                )
            except Exception as e:
                samples.append(
                    {
                        "video_id": str(vid),
                        "error": str(e),
                        "avg_similarity": None,
                        "max_similarity": None,
                        "min_similarity": None,
                    }
                )

    valid_samples = [x for x in samples if x.get("avg_similarity") is not None]
    summary = {
        "num_videos": len(samples),
        "num_valid_videos": len(valid_samples),
        "dataset_avg_similarity_mean": (
            float(np.mean([x["avg_similarity"] for x in valid_samples]))
            if valid_samples
            else None
        ),
        "dataset_max_similarity_mean": (
            float(np.mean([x["max_similarity"] for x in valid_samples]))
            if valid_samples
            else None
        ),
        "dataset_min_similarity_mean": (
            float(np.mean([x["min_similarity"] for x in valid_samples]))
            if valid_samples
            else None
        ),
    }

    return {
        "dataset": dataset_tag,
        "feature_path": feature_path,
        "summary": summary,
        "samples": samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze cosine similarity statistics of video features."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        default="both",
        help=(
            "activitynet | charades | both | "
            "comma-separated values (e.g. activitynet,charades)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="video_feature_similarity.json",
        help="Output json path (default: current directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = parse_datasets_arg(args.datasets)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)

    results = []
    for ds in selected:
        results.append(analyze_one_dataset(ds))

    package = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "project_root": PROJECT_ROOT,
            "datasets": selected,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved analysis json to: {output_path}")


if __name__ == "__main__":
    main()
