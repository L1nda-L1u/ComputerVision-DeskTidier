"""
Rule-Based Language Recommendation Module

Takes Detection objects + tidy_score result and produces
human-readable tidying recommendations. No GPT / API needed.

Usage:
    from desk_language_recommend import generate_language_recommendations

    result = tidy_score(detections, image_size)
    rec = generate_language_recommendations(detections, result)
    print(rec["decision"])
    for s in rec["suggestions"]:
        print(f"  - {s}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ──────────────────── Label → recommendation mapping ──────────────────────────

_ZONE_MAP: Dict[str, Tuple[str, str]] = {
    # label -> (zone_name, action_verb)
    "laptop":         ("workspace",      "Keep"),
    "mouse":          ("workspace",      "Keep"),
    "keyboard":       ("workspace",      "Keep"),
    "clock":          ("workspace",      "Keep"),
    "remote":         ("workspace",      "Keep"),

    "book":           ("reference zone",   "Move"),
    "notebook":       ("reference zone",   "Move"),
    "sticky note":    ("reference zone",   "Move"),
    "paper":          ("reference zone",   "Move"),

    "pen":            ("stationery zone",  "Move"),
    "pencil":         ("stationery zone",  "Move"),
    "marker":         ("stationery zone",  "Move"),
    "eraser":         ("stationery zone",  "Move"),
    "scissors":       ("stationery zone",  "Move"),
    "scissor":        ("stationery zone",  "Move"),
    "tape":           ("stationery zone",  "Move"),
    "phone":          ("stationery zone",  "Move"),
    "cell phone":     ("stationery zone",  "Move"),
    "earphones":      ("stationery zone",  "Move"),

    "cup":            ("temporary item zone", "Move"),
    "mug":            ("temporary item zone", "Move"),
    "coffee cup":     ("temporary item zone", "Move"),
    "bowl":           ("temporary item zone", "Move"),
    "bottle":         ("temporary item zone", "Move"),
    "ring-pull can":  ("temporary item zone", "Move"),

    "cable":          ("desk edge",       "Route"),
    "charger":        ("desk edge",       "Route"),

    "spitball":       ("off the desk",    "Remove"),
    "food packaging": ("off the desk",    "Remove"),
    "trash":          ("off the desk",    "Remove"),
    "packaging":      ("off the desk",    "Remove"),
    "food":           ("off the desk",    "Remove"),
    "dining table":   ("off the desk",    "Remove"),
}


def recommendation_for_label(label: str) -> Optional[str]:
    """
    Return a single actionable sentence for a detected object label.
    Returns None if no specific recommendation applies.
    """
    key = label.strip().lower()
    if key not in _ZONE_MAP:
        return None

    zone, verb = _ZONE_MAP[key]

    if verb == "Keep":
        return f"Keep the {key} in the {zone}."
    if verb == "Remove":
        return f"Remove {key} from the desk."
    if verb == "Route":
        return f"Route the {key} towards the {zone}."
    return f"Move {key} to the {zone}."


# ──────────────────── Global reason generators ────────────────────────────────

def _global_reasons(penalties: Dict[str, int], dispersion_label: str) -> List[str]:
    """Generate high-level reasons from penalty values."""
    reasons: List[str] = []

    if penalties.get("workspace_obstruction_penalty", 0) > 0:
        reasons.append("The central workspace is obstructed by non-essential items.")

    if penalties.get("object_load_penalty", 0) >= 10:
        reasons.append("Too many objects are on the desk.")
    elif penalties.get("object_load_penalty", 0) > 0:
        reasons.append("The desk has a moderate number of objects.")

    if penalties.get("category_penalty", 0) > 0:
        reasons.append("Several temporary or clutter-related items are present.")

    if penalties.get("spatial_overlap_penalty", 0) > 0:
        reasons.append("Some objects are overlapping or stacked.")

    if dispersion_label in ("Medium", "High"):
        reasons.append("Objects are scattered across the desk.")

    if penalties.get("alignment_penalty", 0) > 0:
        reasons.append("Some objects are misaligned with the desk orientation.")

    return reasons


# ──────────────────── Strategic suggestions ───────────────────────────────────

def _strategic_suggestions(
    penalties: Dict[str, int],
    dispersion_label: str,
    cat_labels: Dict[str, List[str]],
) -> List[str]:
    """
    Generate high-level strategic suggestions (before per-object ones).
    cat_labels: {"Temporary Items": ["cup", "bowl"], ...}
    """
    suggestions: List[str] = []

    if penalties.get("workspace_obstruction_penalty", 0) > 0:
        suggestions.append("Clear the central workspace first.")

    if cat_labels.get("Clutter Items"):
        items = ", ".join(sorted(set(cat_labels["Clutter Items"])))
        suggestions.append(f"Remove clutter items from the desk ({items}).")

    if cat_labels.get("Temporary Items"):
        suggestions.append("Relocate temporary items to the temporary item zone.")

    if dispersion_label in ("Medium", "High"):
        suggestions.append("Regroup scattered items into functional zones.")

    if penalties.get("spatial_overlap_penalty", 0) > 0:
        suggestions.append("Separate overlapping items to reduce visual clutter.")

    if penalties.get("alignment_penalty", 0) > 0:
        suggestions.append("Align objects parallel to the desk edges.")

    return suggestions


# ──────────────────── Main entry point ────────────────────────────────────────

def generate_language_recommendations(
    detections: list,
    tidy_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate rule-based language recommendations from detections and tidy_score output.

    Parameters
    ----------
    detections : list[Detection]
        The Detection objects produced by YOLO + scoring pipeline.
    tidy_result : dict
        The dict returned by tidy_score().

    Returns
    -------
    dict with keys: decision, tidy_score, tidy_level, reasons, suggestions
    """
    score = tidy_result["tidy_score"]
    level = tidy_result["tidy_level"]
    penalties = tidy_result["penalties"]
    dispersion_label = tidy_result["explanation"].get("dispersion_label", "Low")

    # Decision
    if score >= 85:
        decision = "No immediate tidying needed."
    elif score >= 70:
        decision = "Minor tidying recommended."
    else:
        decision = "Tidying needed."

    # Categorise detected labels for strategic suggestions
    # Import infer_category from scoring system
    try:
        import importlib.util
        import sys
        from pathlib import Path

        _scoring_path = Path(__file__).resolve().parent / "scoring_module" / "scripts" / "Tidy Scoring System.py"
        _spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(_scoring_path))
        if _spec is None or _spec.loader is None:
            raise RuntimeError("Failed to load scoring module")
        _scoring = importlib.util.module_from_spec(_spec)
        sys.modules["tidy_scoring_system"] = _scoring
        _spec.loader.exec_module(_scoring)
        _infer = _scoring.infer_category
    except Exception:
        def _infer(label):
            return "Temporary Items"

    cat_labels: Dict[str, List[str]] = {}
    for d in detections:
        cat = _infer(d.label)
        cat_labels.setdefault(cat, []).append(d.label)

    # Global reasons
    reasons = _global_reasons(penalties, dispersion_label)
    if not reasons:
        reasons.append("The desk is generally tidy.")

    # Strategic suggestions
    suggestions = _strategic_suggestions(penalties, dispersion_label, cat_labels)

    # Per-object suggestions (deduplicated, grouped by target zone)
    seen_labels = set()
    per_object: List[str] = []
    for d in detections:
        key = d.label.strip().lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        rec = recommendation_for_label(d.label)
        if rec:
            per_object.append(rec)

    suggestions.extend(per_object)

    return {
        "decision": decision,
        "tidy_score": score,
        "tidy_level": level,
        "reasons": reasons,
        "suggestions": suggestions,
    }


# ──────────────────── Standalone demo / test ──────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

    import importlib.util
    from pathlib import Path

    _scoring_path = Path(__file__).resolve().parent / "scoring_module" / "scripts" / "Tidy Scoring System.py"
    _spec = importlib.util.spec_from_file_location("tidy_scoring_system", str(_scoring_path))
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Failed to load scoring module")
    scoring = importlib.util.module_from_spec(_spec)
    import sys as _sys
    _sys.modules["tidy_scoring_system"] = scoring
    _spec.loader.exec_module(scoring)
    Detection = scoring.Detection

    # Demo detections (a messy desk)
    demo = [
        Detection("laptop",   0.92, (250, 140, 430, 250)),
        Detection("mouse",    0.80, (440, 180, 490, 220)),
        Detection("pen",      0.75, (200, 200, 220, 260)),
        Detection("marker",   0.70, (150, 210, 170, 270)),
        Detection("cup",      0.65, (300, 60, 350, 120)),
        Detection("bowl",     0.55, (100, 170, 180, 250)),
        Detection("bottle",   0.60, (500, 100, 540, 180)),
        Detection("spitball", 0.45, (320, 280, 360, 320)),
        Detection("cable",    0.50, (50, 300, 200, 340)),
        Detection("scissors", 0.55, (420, 260, 460, 300)),
        Detection("book",     0.70, (10, 50, 100, 130)),
    ]

    result = scoring.tidy_score(demo, image_size=(640, 360))
    rec = generate_language_recommendations(demo, result)

    print(f"\nTidy Score: {rec['tidy_score']}  ({rec['tidy_level']})")
    print(f"Decision:   {rec['decision']}")
    print("\nReasons:")
    for r in rec["reasons"]:
        print(f"  - {r}")
    print("\nSuggestions:")
    for s in rec["suggestions"]:
        print(f"  - {s}")
