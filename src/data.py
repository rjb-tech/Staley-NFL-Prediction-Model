from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.timer import Timer

_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"


@dataclass
class TeamStats:
    """Per-team aggregated stats for a window of games."""

    # Offensive efficiency
    off_rush_epa: float
    off_pass_epa: float

    # Defensive efficiency
    def_rush_epa: float
    def_pass_epa: float

    # First down rate
    off_fdr: float
    def_fdr: float

    # Turnovers
    to_differential: float

    # Explosive plays (10+ yard runs, 20+ yard passes)
    off_exp_rate: float
    def_exp_rate: float

    # Line metrics (sacks + QB hits + pressures + hurries, per game)
    ol_metric: float
    dl_metric: float

    # Penalties per game
    penalties: float
    penalty_yards: float


@dataclass
class Matchup:
    """A game between two teams."""

    away: TeamStats
    home: TeamStats
    div: bool


def load_raw_pbp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load play-by-play data from nflfastR.

    Returns a tuple of (training_data, testing_data) where:
    - training_data: all seasons through last year
    - testing_data: the current year's season
    """

    current_year = datetime.now().year

    with Timer("PBP loading"):
        # all seasons of pbp data from 1999 - current_year
        training_data = pd.concat(
            [
                pd.read_parquet(_PBP_URL.format(year=y))
                for y in range(1999, current_year - 1)
            ],
            ignore_index=True,
        )

        # latest season
        testing_data = pd.read_parquet(_PBP_URL.format(year=current_year - 1))

    return training_data, testing_data
