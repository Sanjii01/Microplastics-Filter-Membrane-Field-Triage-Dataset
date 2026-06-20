from pathlib import Path
import hashlib
import math
import struct
import zlib

import numpy as np
import pandas as pd


IMAGE_SIZE = 160
N_SCENES = 480
QUESTION_TYPES = [
    "dominant_particle_family",
    "hotspot_quadrant",
    "fiber_tint",
    "bead_count",
    "clogging_state",
    "qc_action",
]

CHOICES = {
    "dominant_particle_family": ["fibers", "fragments", "beads", "films"],
    "hotspot_quadrant": ["upper left", "upper right", "lower left", "lower right"],
    "fiber_tint": ["blue", "red", "black", "clear or pale"],
    "bead_count": ["0-12", "13-18", "19-24", "25 or more"],
    "clogging_state": [
        "open pores",
        "light edge clogging",
        "central mat clogging",
        "patchy multi-zone clogging",
    ],
    "qc_action": ["count now", "reimage focus", "dilute and refilter", "manual review"],
}

QUESTIONS = {
    "dominant_particle_family": "Which suspicious particle family is most common in this membrane field?",
    "hotspot_quadrant": "Which quadrant contains the densest suspicious-particle hotspot?",
    "fiber_tint": "What tint is most common among the visible fibers?",
    "bead_count": "How many round bead-like microplastic particles are visible?",
    "clogging_state": "Which membrane clogging pattern best describes this field?",
    "qc_action": "What is the best next triage action for this field of view?",
}

FAMILY_COLORS = {
    "fibers": {
        "blue": (42, 96, 176),
        "red": (170, 55, 65),
        "black": (34, 36, 38),
        "clear or pale": (205, 210, 198),
    },
    "fragments": [(45, 80, 155), (170, 70, 60), (60, 65, 66), (225, 218, 176)],
    "beads": [(35, 85, 160), (205, 78, 80), (48, 50, 54), (230, 225, 190)],
    "films": [(78, 130, 190), (190, 95, 100), (92, 92, 90), (230, 230, 202)],
}


def _write_png(path: Path, arr: np.ndarray) -> None:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 1))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _blend(arr, mask, color, alpha):
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(color)


def _disk(arr, cx, cy, r, color, alpha=1.0):
    yy, xx = np.ogrid[: arr.shape[0], : arr.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    _blend(arr, mask, color, alpha)


def _line(arr, p0, p1, width, color, alpha=1.0):
    x0, y0 = p0
    x1, y1 = p1
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1)) + 1
    for t in np.linspace(0, 1, steps):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _disk(arr, x, y, max(1, width), color, alpha)


def _rect(arr, x0, y0, x1, y1, color, alpha=1.0):
    x0, x1 = sorted((max(0, x0), min(arr.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(arr.shape[0], y1)))
    if x1 > x0 and y1 > y0:
        arr[y0:y1, x0:x1] = (1 - alpha) * arr[y0:y1, x0:x1] + alpha * np.array(color)


def _quadrant(x, y):
    if y < IMAGE_SIZE / 2 and x < IMAGE_SIZE / 2:
        return "upper_left"
    if y < IMAGE_SIZE / 2 and x >= IMAGE_SIZE / 2:
        return "upper_right"
    if y >= IMAGE_SIZE / 2 and x < IMAGE_SIZE / 2:
        return "lower_left"
    return "lower_right"


def _quadrant_label(q):
    return {"upper_left": "A", "upper_right": "B", "lower_left": "C", "lower_right": "D"}[q]


def _choice_label(question_type, answer_text):
    return "ABCD"[CHOICES[question_type].index(answer_text)]


def _bead_bin(n):
    if n <= 12:
        return "0-12"
    if n <= 18:
        return "13-18"
    if n <= 24:
        return "19-24"
    return "25 or more"


def _draw_membrane(rng, visibility, clogging_state):
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
    tone = rng.choice([(224, 226, 218), (214, 220, 212), (230, 224, 211), (218, 218, 222)])
    base[:] = np.array(tone) + rng.normal(0, 4, base.shape)

    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    cx, cy = IMAGE_SIZE / 2 + rng.normal(0, 2), IMAGE_SIZE / 2 + rng.normal(0, 2)
    membrane = (xx - cx) ** 2 + (yy - cy) ** 2 <= 73**2
    base[~membrane] *= 0.72
    _disk(base, int(cx), int(cy), 75, (238, 238, 228), 0.12)

    spacing = rng.integers(10, 14)
    pore_color = (128, 134, 132)
    for y in range(13, IMAGE_SIZE - 12, spacing):
        for x in range(13, IMAGE_SIZE - 12, spacing):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 70**2 and rng.random() > 0.08:
                r = int(rng.integers(1, 3))
                _disk(base, x + int(rng.normal(0, 1)), y + int(rng.normal(0, 1)), r, pore_color, rng.uniform(0.22, 0.46))

    if clogging_state == "light edge clogging":
        for _ in range(rng.integers(4, 7)):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(53, 70)
            x, y = int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))
            _disk(base, x, y, rng.integers(10, 18), (112, 104, 83), rng.uniform(0.18, 0.32))
    elif clogging_state == "central mat clogging":
        for _ in range(rng.integers(9, 14)):
            x, y = int(cx + rng.normal(0, 17)), int(cy + rng.normal(0, 17))
            _disk(base, x, y, rng.integers(11, 23), (116, 108, 88), rng.uniform(0.18, 0.36))
    elif clogging_state == "patchy multi-zone clogging":
        for _ in range(rng.integers(8, 15)):
            x, y = int(rng.uniform(24, 136)), int(rng.uniform(24, 136))
            _disk(base, x, y, rng.integers(8, 18), (122, 111, 86), rng.uniform(0.16, 0.34))

    if visibility == "low_contrast":
        base = base * 0.84 + 31
    elif visibility == "glare":
        for _ in range(4):
            _disk(base, rng.integers(22, 138), rng.integers(20, 140), rng.integers(12, 26), (248, 246, 230), 0.38)
    elif visibility == "biofilm":
        for _ in range(12):
            _disk(base, rng.integers(10, 150), rng.integers(10, 150), rng.integers(6, 14), (185, 202, 168), 0.18)
    elif visibility == "overloaded":
        base += rng.normal(0, 7, base.shape)

    return base


def _draw_fiber(arr, rng, x, y, tint, difficulty):
    color = FAMILY_COLORS["fibers"][tint]
    length = rng.uniform(18, 48 if difficulty != "easy" else 34)
    theta = rng.uniform(0, 2 * math.pi)
    curve = rng.normal(0, 0.35)
    width = int(rng.integers(1, 3))
    prev = (x, y)
    for i in range(1, 5):
        t = i / 4
        px = x + length * (t - 0.5) * math.cos(theta + curve * t)
        py = y + length * (t - 0.5) * math.sin(theta + curve * t)
        cur = (int(px), int(py))
        _line(arr, prev, cur, width, color, 0.78)
        prev = cur


def _draw_fragment(arr, rng, x, y, visibility):
    color = FAMILY_COLORS["fragments"][int(rng.integers(0, 4))]
    w = int(rng.integers(4, 11))
    h = int(rng.integers(3, 10))
    _rect(arr, x - w, y - h // 2, x + w, y + h // 2, color, 0.72)
    _disk(arr, x + int(rng.normal(0, 3)), y + int(rng.normal(0, 3)), rng.integers(2, 5), color, 0.78)
    if visibility == "low_contrast":
        _disk(arr, x, y, max(w, h), (230, 230, 220), 0.15)


def _draw_bead(arr, rng, x, y):
    color = FAMILY_COLORS["beads"][int(rng.integers(0, 4))]
    r = int(rng.integers(3, 6))
    _disk(arr, x, y, r + 1, (40, 42, 42), 0.18)
    _disk(arr, x, y, r, color, 0.8)
    _disk(arr, x - max(1, r // 2), y - max(1, r // 2), max(1, r // 3), (250, 248, 232), 0.45)


def _draw_film(arr, rng, x, y):
    color = FAMILY_COLORS["films"][int(rng.integers(0, 4))]
    for _ in range(rng.integers(3, 7)):
        _disk(
            arr,
            x + int(rng.normal(0, 5)),
            y + int(rng.normal(0, 5)),
            rng.integers(5, 12),
            color,
            rng.uniform(0.18, 0.35),
        )


def _make_scene(rng, scene_index):
    difficulty = rng.choice(["easy", "medium", "hard"], p=[0.28, 0.47, 0.25])
    visibility = rng.choice(["clear", "low_contrast", "glare", "biofilm", "overloaded"], p=[0.38, 0.16, 0.14, 0.14, 0.18])
    layout_family = rng.choice(["centered", "edge_loaded", "ring_bias", "diagonal_streak"], p=[0.46, 0.20, 0.18, 0.16])
    ood_axis = rng.choice(["standard", "blue_stain", "edge_shadow", "bubble_ring"], p=[0.62, 0.14, 0.13, 0.11])
    clogging_state = rng.choice(
        ["open pores", "light edge clogging", "central mat clogging", "patchy multi-zone clogging"],
        p=[0.34, 0.24, 0.20, 0.22],
    )
    arr = _draw_membrane(rng, visibility, clogging_state)

    if difficulty == "easy":
        n_particles = int(rng.integers(24, 44))
    elif difficulty == "medium":
        n_particles = int(rng.integers(48, 76))
    else:
        n_particles = int(rng.integers(78, 122))

    dominant = rng.choice(["fibers", "fragments", "beads", "films"])
    weights = np.array([0.18, 0.18, 0.18, 0.18])
    family_order = ["fibers", "fragments", "beads", "films"]
    weights[family_order.index(dominant)] = rng.uniform(0.36, 0.52)
    weights /= weights.sum()
    family_counts = {f: 0 for f in family_order}
    quadrant_weights = {"upper_left": 0, "upper_right": 0, "lower_left": 0, "lower_right": 0}
    fiber_tints = {"blue": 0, "red": 0, "black": 0, "clear or pale": 0}
    bead_count = 0

    hotspot = rng.choice(list(quadrant_weights))
    for _ in range(n_particles):
        if layout_family == "edge_loaded" and rng.random() < 0.55:
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(48, 70)
            x = int(IMAGE_SIZE / 2 + rad * math.cos(ang))
            y = int(IMAGE_SIZE / 2 + rad * math.sin(ang))
        elif layout_family == "ring_bias" and rng.random() < 0.45:
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.normal(47, 8)
            x = int(IMAGE_SIZE / 2 + rad * math.cos(ang))
            y = int(IMAGE_SIZE / 2 + rad * math.sin(ang))
        elif layout_family == "diagonal_streak" and rng.random() < 0.45:
            x = int(rng.uniform(20, 140))
            y = int(np.clip(x + rng.normal(0, 17), 15, 145))
        elif rng.random() < 0.34:
            x = int(rng.uniform(10, 70) if "left" in hotspot else rng.uniform(90, 150))
            y = int(rng.uniform(10, 70) if "upper" in hotspot else rng.uniform(90, 150))
        else:
            x = int(rng.uniform(12, 148))
            y = int(rng.uniform(12, 148))

        family = rng.choice(family_order, p=weights)
        family_counts[family] += 1
        quadrant_weights[_quadrant(x, y)] += 1
        if family == "fibers":
            tint = rng.choice(["blue", "red", "black", "clear or pale"], p=[0.30, 0.22, 0.25, 0.23])
            fiber_tints[tint] += 1
            _draw_fiber(arr, rng, x, y, tint, difficulty)
        elif family == "fragments":
            _draw_fragment(arr, rng, x, y, visibility)
        elif family == "beads":
            bead_count += 1
            _draw_bead(arr, rng, x, y)
        else:
            _draw_film(arr, rng, x, y)

    if ood_axis == "blue_stain":
        arr[:, :, 2] += 18
        arr[:, :, 0] -= 8
    elif ood_axis == "edge_shadow":
        yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        shadow = np.clip((xx + yy) / (2 * IMAGE_SIZE), 0, 1)
        arr *= 0.82 + 0.22 * shadow[..., None]
    elif ood_axis == "bubble_ring":
        for r, alpha in [(60, 0.22), (63, 0.14), (66, 0.10)]:
            yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            ring = np.abs((xx - 80) ** 2 + (yy - 80) ** 2 - r**2) < 55
            _blend(arr, ring, (250, 250, 232), alpha)

    arr += rng.normal(0, 5 if difficulty != "hard" else 8, arr.shape)

    if visibility in ["low_contrast", "glare"]:
        qc_action = "reimage focus"
    elif visibility == "overloaded":
        qc_action = "dilute and refilter"
    elif clogging_state in ["central mat clogging", "patchy multi-zone clogging"] and n_particles > 75:
        qc_action = "dilute and refilter"
    elif visibility == "biofilm" or ood_axis in ["edge_shadow", "bubble_ring"]:
        qc_action = "manual review"
    else:
        qc_action = "count now"

    dominant_family = max(family_counts, key=family_counts.get)
    hotspot_quadrant = max(quadrant_weights, key=quadrant_weights.get)
    fiber_tint = max(fiber_tints, key=fiber_tints.get) if sum(fiber_tints.values()) else "clear or pale"

    return arr, {
        "difficulty": difficulty,
        "visibility": visibility,
        "layout_family": layout_family,
        "ood_axis": ood_axis,
        "dominant_particle_family": dominant_family,
        "hotspot_quadrant": hotspot_quadrant,
        "fiber_tint": fiber_tint,
        "bead_count": bead_count,
        "bead_count_bin": _bead_bin(bead_count),
        "clogging_state": clogging_state,
        "qc_action": qc_action,
        "density_index": n_particles,
    }


def _answer_for(question_type, trace):
    if question_type == "dominant_particle_family":
        return _choice_label(question_type, trace["dominant_particle_family"])
    if question_type == "hotspot_quadrant":
        return _quadrant_label(trace["hotspot_quadrant"])
    if question_type == "fiber_tint":
        return _choice_label(question_type, trace["fiber_tint"])
    if question_type == "bead_count":
        return _choice_label(question_type, trace["bead_count_bin"])
    if question_type == "clogging_state":
        return _choice_label(question_type, trace["clogging_state"])
    if question_type == "qc_action":
        return _choice_label(question_type, trace["qc_action"])
    raise ValueError(question_type)


def main():
    root = Path(__file__).resolve().parent
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for old_png in image_dir.glob("*.png"):
        old_png.unlink()
    rng = np.random.default_rng(20260620)
    rows = []

    for scene_index in range(N_SCENES):
        arr, trace = _make_scene(rng, scene_index)
        scene_id = f"mem_scene_{scene_index:05d}"
        image_name = f"{scene_id}.png"
        _write_png(image_dir / image_name, arr)
        for question_type in QUESTION_TYPES:
            qid_seed = f"{scene_id}:{question_type}".encode()
            qid = "mpq_" + hashlib.sha1(qid_seed).hexdigest()[:14]
            choices = CHOICES[question_type]
            rows.append(
                {
                    "question_id": qid,
                    "scene_id": scene_id,
                    "image_path": f"images/{image_name}",
                    "question_type": question_type,
                    "question": QUESTIONS[question_type],
                    "choice_a": choices[0],
                    "choice_b": choices[1],
                    "choice_c": choices[2],
                    "choice_d": choices[3],
                    "answer_label": _answer_for(question_type, trace),
                    "difficulty": trace["difficulty"],
                    "visibility": trace["visibility"],
                    "layout_family": trace["layout_family"],
                    "ood_axis": trace["ood_axis"],
                    "trace_dominant_particle_family": trace["dominant_particle_family"],
                    "trace_hotspot_quadrant": trace["hotspot_quadrant"],
                    "trace_fiber_tint": trace["fiber_tint"],
                    "trace_bead_count": trace["bead_count"],
                    "trace_bead_count_bin": trace["bead_count_bin"],
                    "trace_clogging_state": trace["clogging_state"],
                    "trace_qc_action": trace["qc_action"],
                    "trace_density_index": trace["density_index"],
                }
            )

    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    print(f"wrote {len(rows)} rows and {N_SCENES} images to {root}")


if __name__ == "__main__":
    main()
