"""
Team name normalization for CBB Totals Model.
Resolves variant team names (ESPN, Torvik, Odds API) to a single canonical name.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "team_name_mapping.yaml"
)


class TeamNormalizer:
    """
    Resolves raw team name strings to canonical names.

    The mapping YAML file uses the following structure::

        canonical_name:
          - variant1
          - variant2
          ...

    Usage::

        normalizer = TeamNormalizer(config)
        canonical = normalizer.normalize("UConn")     # → "Connecticut"
        team_id   = normalizer.get_id("North Carolina")  # → "north_carolina"

    Fuzzy matching (difflib) is used as a fallback when an exact match
    is not found, with a configurable similarity threshold.
    """

    def __init__(
        self,
        config: dict = None,
        mapping_path: Optional[str] = None,
        fuzzy_threshold: float = 0.85,
    ):
        self.config = config or {}
        self.fuzzy_threshold = fuzzy_threshold

        # Resolve mapping file path
        if mapping_path:
            self._mapping_path = Path(mapping_path)
        else:
            self._mapping_path = _DEFAULT_MAPPING_PATH

        # Internal dicts
        # variant_name_lower → canonical_name
        self._variant_to_canonical: dict[str, str] = {}
        # canonical_name_lower → canonical_name (for round-trip lookups)
        self._canonical_set: dict[str, str] = {}

        self._load_mapping()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load_mapping(self) -> None:
        """Load the YAML mapping file and build lookup structures."""
        if not self._mapping_path.exists():
            logger.warning(
                f"Team name mapping file not found: {self._mapping_path}. "
                "Using empty mapping."
            )
            return

        try:
            with open(self._mapping_path, "r", encoding="utf-8") as f:
                raw: dict = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error(f"Failed to load team name mapping: {exc}")
            return

        count = 0
        for canonical, variants in raw.items():
            canonical = str(canonical).strip()
            canonical_lower = canonical.lower()

            # Canonical name maps to itself
            self._variant_to_canonical[canonical_lower] = canonical
            self._canonical_set[canonical_lower] = canonical

            if not isinstance(variants, list):
                continue

            for variant in variants:
                variant_str = str(variant).strip()
                variant_lower = variant_str.lower()
                if variant_lower not in self._variant_to_canonical:
                    self._variant_to_canonical[variant_lower] = canonical
                count += 1

        logger.debug(
            f"TeamNormalizer loaded {len(self._canonical_set)} canonical teams "
            f"with {count} variant mappings."
        )

    # ── Core methods ──────────────────────────────────────────────────────

    def normalize(self, name: str) -> str:
        """
        Resolve a raw team name to its canonical form.

        Lookup order:
        1. Exact match (case-insensitive)
        2. Fuzzy match via difflib

        Args:
            name: Raw team name string.

        Returns:
            Canonical team name, or the original name if no match found.
        """
        if not name or not isinstance(name, str):
            return name or ""

        name_stripped = name.strip()
        name_lower = name_stripped.lower()

        # Exact match
        if name_lower in self._variant_to_canonical:
            return self._variant_to_canonical[name_lower]

        # Fuzzy match
        best_match = self._fuzzy_match(name_lower)
        if best_match:
            canonical = self._variant_to_canonical[best_match]
            logger.debug(
                f"TeamNormalizer fuzzy match: '{name}' → '{canonical}' "
                f"(via '{best_match}')"
            )
            return canonical

        logger.warning(
            f"TeamNormalizer: no match found for '{name}'. Returning as-is."
        )
        return name_stripped

    def get_id(self, name: str) -> Optional[str]:
        """
        Return a canonical team ID (slug) for the given name.

        The ID is the canonical name lowercased with spaces replaced by underscores.

        Args:
            name: Raw or canonical team name.

        Returns:
            Team ID string or None if not recognized.
        """
        canonical = self.normalize(name)
        if canonical == name and name.lower() not in self._variant_to_canonical:
            return None
        return canonical.lower().replace(" ", "_").replace(".", "").replace("'", "")

    def is_known(self, name: str) -> bool:
        """Return True if the name (or its fuzzy match) is in the mapping."""
        name_lower = name.strip().lower()
        if name_lower in self._variant_to_canonical:
            return True
        return bool(self._fuzzy_match(name_lower))

    def get_canonical_names(self) -> list[str]:
        """Return sorted list of all canonical team names."""
        return sorted(self._canonical_set.values())

    # ── Fuzzy matching ────────────────────────────────────────────────────

    def _fuzzy_match(self, name_lower: str) -> Optional[str]:
        """
        Find the best matching variant key using difflib.

        Args:
            name_lower: Lowercase input string.

        Returns:
            The best matching variant key from the lookup dict,
            or None if no match exceeds the threshold.
        """
        candidates = list(self._variant_to_canonical.keys())
        if not candidates:
            return None

        matches = difflib.get_close_matches(
            name_lower,
            candidates,
            n=1,
            cutoff=self.fuzzy_threshold,
        )
        return matches[0] if matches else None

    # ── Runtime mutation ──────────────────────────────────────────────────

    def add_mapping(self, raw_name: str, canonical_name: str) -> None:
        """
        Add or update a mapping at runtime (not persisted to disk).

        Args:
            raw_name:       The variant/alias to map from.
            canonical_name: The canonical team name to map to.
        """
        raw_lower = raw_name.strip().lower()
        canonical_lower = canonical_name.strip().lower()

        self._variant_to_canonical[raw_lower] = canonical_name.strip()
        self._canonical_set[canonical_lower] = canonical_name.strip()

        logger.debug(f"TeamNormalizer: added mapping '{raw_name}' → '{canonical_name}'")

    def match_game(
        self,
        home_name: str,
        away_name: str,
        odds_home: str,
        odds_away: str,
        threshold: float = 0.80,
    ) -> bool:
        """
        Determine if an ESPN game and an Odds API game refer to the same matchup.

        Normalizes all four names and checks pairwise.

        Args:
            home_name:  ESPN home team name.
            away_name:  ESPN away team name.
            odds_home:  Odds API home team name.
            odds_away:  Odds API away team name.
            threshold:  Fuzzy similarity cutoff for last-resort name matching.

        Returns:
            True if both home and away teams match.
        """
        norm_home = self.normalize(home_name).lower()
        norm_away = self.normalize(away_name).lower()
        norm_odds_home = self.normalize(odds_home).lower()
        norm_odds_away = self.normalize(odds_away).lower()

        home_match = (norm_home == norm_odds_home) or _similarity(
            norm_home, norm_odds_home
        ) >= threshold
        away_match = (norm_away == norm_odds_away) or _similarity(
            norm_away, norm_odds_away
        ) >= threshold

        return home_match and away_match

    def find_best_odds_match(
        self,
        espn_home: str,
        espn_away: str,
        odds_games: list[dict],
    ) -> Optional[dict]:
        """
        Find the best matching odds entry for an ESPN game.

        Args:
            espn_home:  ESPN home team name.
            espn_away:  ESPN away team name.
            odds_games: List of odds dicts (each must have 'home_team', 'away_team').

        Returns:
            Best matching odds dict or None.
        """
        norm_home = self.normalize(espn_home).lower()
        norm_away = self.normalize(espn_away).lower()

        best_score = 0.0
        best_match = None

        for odds_game in odds_games:
            oh = self.normalize(odds_game.get("home_team", "")).lower()
            oa = self.normalize(odds_game.get("away_team", "")).lower()

            home_sim = _similarity(norm_home, oh)
            away_sim = _similarity(norm_away, oa)
            combined = (home_sim + away_sim) / 2

            if combined > best_score and combined >= 0.75:
                best_score = combined
                best_match = odds_game

        if best_match:
            logger.debug(
                f"Matched '{espn_home}' vs '{espn_away}' to odds game "
                f"'{best_match.get('home_team')}' vs '{best_match.get('away_team')}' "
                f"(score={best_score:.2f})"
            )
        else:
            logger.debug(
                f"No odds match found for ESPN game: '{espn_home}' vs '{espn_away}'"
            )

        return best_match


# ── Utilities ─────────────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """Return difflib SequenceMatcher ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()
