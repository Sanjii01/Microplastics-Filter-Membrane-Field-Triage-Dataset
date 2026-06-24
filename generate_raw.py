from pathlib import Path
import hashlib
import math
import struct
import zlib

import numpy as np
import pandas as pd


IMAGE_SIZE = 224
N_SCENES = 360
MEMBRANE_RADIUS = 100

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

FAMILY_ORDER = ["fibers", "fragments", "beads", "films"]
QUADRANTS = ["upper_left", "upper_right", "lower_left", "lower_right"]
FIBER_TINTS = ["blue", "red", "black", "clear or pale"]
CLOGGING_STATES = [
    "open pores",
    "light edge clogging",
    "central mat clogging",
    "patchy multi-zone clogging",
]
QC_MODES = ["count now", "reimage focus", "dilute and refilter", "manual review"]
BEAD_BINS = ["0-12", "13-18", "19-24", "25 or more"]

FAMILY_COLORS = {
    "fibers": {
        "blue": (38, 92, 190),
        "red": (185, 45, 55),
        "black": (26, 28, 30),
        "clear or pale": (224, 222, 188),
    },
    "fragments": [(34, 70, 170), (200, 56, 64), (44, 48, 50), (232, 205, 120)],
    "beads": [(36, 88, 190), (215, 60, 72), (32, 34, 38), (238, 216, 130)],
    "films": [(82, 146, 205), (206, 105, 114), (96, 100, 96), (238, 235, 194)],
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
    x0 = max(0, int(cx - r - 1))
    x1 = min(arr.shape[1], int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(arr.shape[0], int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    patch = arr[y0:y1, x0:x1]
    patch[mask] = (1 - alpha) * patch[mask] + alpha * np.array(color)


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


def _ring(arr, cx, cy, inner_r, outer_r, color, alpha=1.0):
    x0 = max(0, int(cx - outer_r - 1))
    x1 = min(arr.shape[1], int(cx + outer_r + 2))
    y0 = max(0, int(cy - outer_r - 1))
    y1 = min(arr.shape[0], int(cy + outer_r + 2))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 >= inner_r**2) & (dist2 <= outer_r**2)
    patch = arr[y0:y1, x0:x1]
    patch[mask] = (1 - alpha) * patch[mask] + alpha * np.array(color)


def _quadrant(x, y):
    mid = IMAGE_SIZE / 2
    if y < mid and x < mid:
        return "upper_left"
    if y < mid and x >= mid:
        return "upper_right"
    if y >= mid and x < mid:
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


def _position(rng, hotspot):
    if rng.random() < 0.64:
        x = rng.uniform(24, 96) if "left" in hotspot else rng.uniform(128, 200)
        y = rng.uniform(24, 96) if "upper" in hotspot else rng.uniform(128, 200)
    else:
        angle = rng.uniform(0, 2 * math.pi)
        radius = MEMBRANE_RADIUS * math.sqrt(rng.uniform(0.02, 0.92))
        x = IMAGE_SIZE / 2 + radius * math.cos(angle)
        y = IMAGE_SIZE / 2 + radius * math.sin(angle)
    return int(np.clip(x, 12, IMAGE_SIZE - 13)), int(np.clip(y, 12, IMAGE_SIZE - 13))


def _draw_membrane(rng, visibility, clogging_state, ood_axis):
    arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
    tone = rng.choice([(232, 234, 224), (224, 231, 222), (235, 229, 216), (226, 226, 232)])
    arr[:] = np.array(tone) + rng.normal(0, 3.0, arr.shape)

    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    cx = IMAGE_SIZE / 2 + rng.normal(0, 1.5)
    cy = IMAGE_SIZE / 2 + rng.normal(0, 1.5)
    membrane = (xx - cx) ** 2 + (yy - cy) ** 2 <= MEMBRANE_RADIUS**2
    arr[~membrane] *= 0.62
    _disk(arr, int(cx), int(cy), MEMBRANE_RADIUS + 2, (246, 246, 235), 0.11)

    pore_color = (112, 120, 118)
    for y in range(18, IMAGE_SIZE - 17, 12):
        for x in range(18, IMAGE_SIZE - 17, 12):
            if (x - cx) ** 2 + (y - cy) ** 2 <= (MEMBRANE_RADIUS - 6) ** 2 and rng.random() > 0.05:
                _disk(arr, x + int(rng.normal(0, 1)), y + int(rng.normal(0, 1)), 2, pore_color, 0.38)

    if clogging_state == "light edge clogging":
        _ring(arr, int(cx), int(cy), 76, 99, (110, 96, 70), 0.18)
        for _ in range(10):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(74, 96)
            _disk(
                arr,
                int(cx + rad * math.cos(ang)),
                int(cy + rad * math.sin(ang)),
                rng.integers(9, 18),
                (95, 82, 58),
                rng.uniform(0.20, 0.34),
            )
    elif clogging_state == "central mat clogging":
        _disk(arr, int(cx), int(cy), 44, (105, 92, 70), 0.35)
        _disk(arr, int(cx) + 6, int(cy) - 5, 31, (88, 78, 60), 0.25)
        for _ in range(18):
            _disk(
                arr,
                int(cx + rng.normal(0, 26)),
                int(cy + rng.normal(0, 26)),
                rng.integers(7, 15),
                (118, 104, 80),
                rng.uniform(0.18, 0.30),
            )
    elif clogging_state == "patchy multi-zone clogging":
        centers = [(62, 63), (163, 62), (74, 158), (154, 153)]
        for px, py in centers:
            _disk(arr, px + int(rng.normal(0, 5)), py + int(rng.normal(0, 5)), rng.integers(18, 27), (112, 98, 72), 0.30)
        for _ in range(10):
            _disk(arr, rng.integers(28, 198), rng.integers(28, 198), rng.integers(8, 16), (124, 108, 78), 0.16)

    if visibility == "low_contrast":
        arr = arr * 0.72 + 58
    elif visibility == "glare":
        for gx, gy, gr in [(62, 55, 28), (160, 65, 22), (130, 150, 24)]:
            _disk(arr, gx + int(rng.normal(0, 6)), gy + int(rng.normal(0, 6)), gr, (255, 252, 230), 0.46)
    elif visibility == "biofilm":
        for _ in range(18):
            _disk(arr, rng.integers(20, 204), rng.integers(20, 204), rng.integers(8, 21), (170, 204, 153), 0.20)
    elif visibility == "overloaded":
        arr += rng.normal(0, 7, arr.shape)

    if ood_axis == "blue_stain":
        arr[:, :, 2] += 20
        arr[:, :, 0] -= 7
    elif ood_axis == "edge_shadow":
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        shadow = np.clip((dist - 52) / 52, 0, 1)
        arr *= (1.0 - 0.22 * shadow[..., None])
    elif ood_axis == "bubble_ring":
        _ring(arr, int(cx), int(cy), 78, 84, (255, 255, 236), 0.30)
        _ring(arr, int(cx), int(cy), 87, 90, (225, 225, 210), 0.22)

    return arr


def _draw_fiber(arr, rng, x, y, tint, difficulty):
    color = FAMILY_COLORS["fibers"][tint]
    length = rng.uniform(34, 66 if difficulty != "easy" else 52)
    theta = rng.uniform(0, 2 * math.pi)
    bend = rng.normal(0, 0.45)
    width = int(rng.integers(2, 4))
    points = []
    for i in range(6):
        t = i / 5 - 0.5
        px = x + length * t * math.cos(theta + bend * t)
        py = y + length * t * math.sin(theta + bend * t)
        points.append((int(px), int(py)))
    for p0, p1 in zip(points, points[1:]):
        _line(arr, p0, p1, width, color, 0.86)


def _draw_fragment(arr, rng, x, y, visibility):
    color = FAMILY_COLORS["fragments"][int(rng.integers(0, 4))]
    width = int(rng.integers(7, 15))
    height = int(rng.integers(5, 12))
    _rect(arr, x - width, y - height, x + width, y + height // 2, color, 0.76)
    _disk(arr, x - int(width * 0.45), y + int(height * 0.35), rng.integers(4, 8), color, 0.82)
    _disk(arr, x + int(width * 0.35), y - int(height * 0.35), rng.integers(3, 7), color, 0.75)
    if visibility == "low_contrast":
        _disk(arr, x, y, max(width, height), (238, 238, 226), 0.11)


def _draw_bead(arr, rng, x, y):
    color = FAMILY_COLORS["beads"][int(rng.integers(0, 4))]
    radius = int(rng.integers(7, 10))
    _disk(arr, x, y, radius + 2, (34, 36, 38), 0.20)
    _disk(arr, x, y, radius, color, 0.90)
    _disk(arr, x - max(2, radius // 2), y - max(2, radius // 2), max(2, radius // 3), (255, 251, 230), 0.58)


def _draw_film(arr, rng, x, y):
    color = FAMILY_COLORS["films"][int(rng.integers(0, 4))]
    for _ in range(rng.integers(5, 9)):
        _disk(
            arr,
            x + int(rng.normal(0, 8)),
            y + int(rng.normal(0, 8)),
            rng.integers(9, 18),
            color,
            rng.uniform(0.20, 0.36),
        )
    _line(arr, (x - 12, y), (x + 15, y + int(rng.normal(0, 6))), 1, (70, 75, 70), 0.18)


def _planned_counts(scene_index, dominant, bead_count):
    counts = {"fibers": 24, "fragments": 9, "beads": bead_count, "films": 9}
    counts["beads"] = bead_count
    dominant_margin = 26 if scene_index % 3 else 32
    if dominant == "beads":
        counts["beads"] = max(bead_count, 38 + scene_index % 8)
        counts["fibers"] = 22
        counts["fragments"] = 7
        counts["films"] = 7
    elif dominant == "fibers":
        counts["fibers"] = max(bead_count + dominant_margin, 52)
    else:
        counts[dominant] = max(bead_count + dominant_margin, 50)
    return counts


def _scene_modes(scene_index, rng):
    dominant = FAMILY_ORDER[scene_index % 4]
    hotspot = QUADRANTS[(scene_index // 4) % 4]
    fiber_tint = FIBER_TINTS[(scene_index // 16) % 4]
    bead_bin = BEAD_BINS[(scene_index // 64) % 4]
    clogging_state = CLOGGING_STATES[(scene_index // 11) % 4]
    qc_seed = QC_MODES[(scene_index // 7) % 4]

    bead_count = {
        "0-12": int(rng.integers(4, 8)),
        "13-18": int(rng.integers(14, 18)),
        "19-24": int(rng.integers(21, 25)),
        "25 or more": int(rng.integers(34, 43)),
    }[bead_bin]

    difficulty = rng.choice(["easy", "medium", "hard"], p=[0.40, 0.44, 0.16])
    visibility = "clear"
    ood_axis = "standard"
    if qc_seed == "reimage focus":
        visibility = rng.choice(["glare", "low_contrast"], p=[0.56, 0.44])
    elif qc_seed == "dilute and refilter":
        visibility = rng.choice(["overloaded", "clear"], p=[0.62, 0.38])
        if rng.random() < 0.45:
            clogging_state = rng.choice(["central mat clogging", "patchy multi-zone clogging"])
    elif qc_seed == "manual review":
        visibility = rng.choice(["biofilm", "clear"], p=[0.72, 0.28])
        ood_axis = rng.choice(["edge_shadow", "bubble_ring", "standard"], p=[0.36, 0.34, 0.30])
    else:
        if clogging_state in ["central mat clogging", "patchy multi-zone clogging"] and rng.random() < 0.60:
            clogging_state = rng.choice(["open pores", "light edge clogging"])

    if qc_seed != "manual review" and rng.random() < 0.18:
        ood_axis = rng.choice(["standard", "blue_stain"], p=[0.45, 0.55])

    return dominant, hotspot, fiber_tint, bead_count, clogging_state, difficulty, visibility, ood_axis


def _make_scene(rng, scene_index):
    dominant, hotspot, fiber_tint_mode, bead_count, clogging_state, difficulty, visibility, ood_axis = _scene_modes(scene_index, rng)
    arr = _draw_membrane(rng, visibility, clogging_state, ood_axis)
    counts = _planned_counts(scene_index, dominant, bead_count)
    if visibility == "overloaded":
        for family in FAMILY_ORDER:
            counts[family] = int(counts[family] * 1.25)

    family_counts = {family: 0 for family in FAMILY_ORDER}
    quadrant_counts = {quadrant: 0 for quadrant in QUADRANTS}
    fiber_tints = {tint: 0 for tint in FIBER_TINTS}
    particle_order = []
    for family, count in counts.items():
        particle_order.extend([family] * int(count))
    rng.shuffle(particle_order)

    for family in particle_order:
        x, y = _position(rng, hotspot)
        family_counts[family] += 1
        quadrant_counts[_quadrant(x, y)] += 1
        if family == "fibers":
            if rng.random() < 0.88:
                tint = fiber_tint_mode
            else:
                tint = rng.choice(FIBER_TINTS)
            fiber_tints[tint] += 1
            _draw_fiber(arr, rng, x, y, tint, difficulty)
        elif family == "fragments":
            _draw_fragment(arr, rng, x, y, visibility)
        elif family == "films":
            _draw_film(arr, rng, x, y)

    for _ in range(counts["beads"]):
        x, y = _position(rng, hotspot)
        quadrant_counts[_quadrant(x, y)] += 1
        _draw_bead(arr, rng, x, y)

    if visibility == "overloaded":
        for _ in range(34):
            x, y = _position(rng, hotspot)
            _disk(arr, x, y, rng.integers(2, 5), (42, 44, 42), 0.26)

    arr += rng.normal(0, 4 if difficulty != "hard" else 7, arr.shape)

    if visibility in ["glare", "low_contrast"]:
        qc_action = "reimage focus"
    elif visibility == "overloaded" or clogging_state == "central mat clogging":
        qc_action = "dilute and refilter"
    elif visibility == "biofilm" or ood_axis in ["edge_shadow", "bubble_ring"]:
        qc_action = "manual review"
    else:
        qc_action = "count now"

    return arr, {
        "difficulty": difficulty,
        "visibility": visibility,
        "layout_family": "hotspot_" + hotspot,
        "ood_axis": ood_axis,
        "dominant_particle_family": max(family_counts, key=family_counts.get),
        "hotspot_quadrant": max(quadrant_counts, key=quadrant_counts.get),
        "fiber_tint": max(fiber_tints, key=fiber_tints.get) if sum(fiber_tints.values()) else "clear or pale",
        "bead_count": bead_count,
        "bead_count_bin": _bead_bin(bead_count),
        "clogging_state": clogging_state,
        "qc_action": qc_action,
        "density_index": sum(counts.values()),
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
    rng = np.random.default_rng(20260624)
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
