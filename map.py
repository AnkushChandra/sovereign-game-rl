"""
map.py — Map / territory setup for the SOVEREIGN environment.

The default map has 9 territories arranged in a simple graph:
  - 3 home territories (one per nation)
  - 6 contested territories

Layout (conceptual):

    I_HOME ── C1 ── C2 ── D_HOME
               |      |
              C3 ── C4
               |      |
    N_HOME ── C5 ── C6

Adjacency lets the Invader reach the Defender only through contested zones.
"""

from config import NUM_TERRITORIES


# ─────────────────────────────────────────────
# Territory IDs
# ─────────────────────────────────────────────
T_INVADER_HOME = 0
T_C1 = 1
T_C2 = 2
T_DEFENDER_HOME = 3
T_C3 = 4
T_C4 = 5
T_NEUTRAL_HOME = 6
T_C5 = 7
T_C6 = 8

# Readable names for display
TERRITORY_NAMES = {
    0: "Invader Home",
    1: "C1",
    2: "C2",
    3: "Defender Home",
    4: "C3",
    5: "C4",
    6: "Neutral Home",
    7: "C5",
    8: "C6",
}


def build_adjacency():
    """Return adjacency list as dict of sets."""
    adj = {i: set() for i in range(NUM_TERRITORIES)}

    edges = [
        (T_INVADER_HOME, T_C1),
        (T_C1, T_C2),
        (T_C2, T_DEFENDER_HOME),
        (T_C1, T_C3),
        (T_C2, T_C4),
        (T_C3, T_C4),
        (T_C3, T_C5),
        (T_C4, T_C6),
        (T_NEUTRAL_HOME, T_C5),
        (T_C5, T_C6),
    ]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    return adj


def build_default_territories():
    """
    Return a list of territory dicts for the default 9-territory map.

    Each territory has:
        id              : int
        name            : str
        controller      : str   one of "I", "D", "N", "Contested"
        resource_value  : float [0, 1]
        strategic_value : float [0, 1]
        is_home         : str or None  ("I", "D", "N", or None)
    """
    territories = [
        # Home territories
        {"id": 0, "name": "Invader Home",  "controller": "I", "resource_value": 0.3, "strategic_value": 0.5, "is_home": "I"},
        {"id": 3, "name": "Defender Home", "controller": "D", "resource_value": 0.3, "strategic_value": 0.5, "is_home": "D"},
        {"id": 6, "name": "Neutral Home",  "controller": "N", "resource_value": 0.2, "strategic_value": 0.2, "is_home": "N"},
        # Contested territories (higher value toward center)
        {"id": 1, "name": "C1", "controller": "Contested", "resource_value": 0.5, "strategic_value": 0.6, "is_home": None},
        {"id": 2, "name": "C2", "controller": "Contested", "resource_value": 0.5, "strategic_value": 0.6, "is_home": None},
        {"id": 4, "name": "C3", "controller": "Contested", "resource_value": 0.4, "strategic_value": 0.4, "is_home": None},
        {"id": 5, "name": "C4", "controller": "Contested", "resource_value": 0.4, "strategic_value": 0.4, "is_home": None},
        {"id": 7, "name": "C5", "controller": "Contested", "resource_value": 0.3, "strategic_value": 0.3, "is_home": None},
        {"id": 8, "name": "C6", "controller": "Contested", "resource_value": 0.3, "strategic_value": 0.3, "is_home": None},
    ]

    # Sort by id for convenience
    territories.sort(key=lambda t: t["id"])
    return territories


def build_default_map():
    """
    Convenience function that returns (territories, adjacency).

    Returns:
        territories : list[dict]  — length-9 list indexed by territory id
        adjacency   : dict[int, set[int]]
    """
    return build_default_territories(), build_adjacency()
