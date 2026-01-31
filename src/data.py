from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.timer import Timer

_PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
_TEAMS_URL = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.parquet"


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

    # Third-down conversion rate
    off_third_down_pct: float
    def_third_down_pct: float

    # Red zone efficiency (EPA inside the 20)
    off_rz_epa: float
    def_rz_epa: float

    # Win probability added (per play)
    off_wpa: float
    def_wpa: float

    # Passing detail
    off_completion_pct: float
    def_completion_pct: float
    off_air_yards: float
    def_air_yards: float
    off_yac: float
    def_yac: float

    # Pressure rate (pressures / dropbacks)
    off_pressure_rate: float
    def_pressure_rate: float

    # Penalties per game
    penalties: float
    penalty_yards: float


@dataclass
class TeamInfo:
    abbr: str
    name: str
    team_logo_link: str
    division: str
    conference: str
    color_1: str
    color_2: str
    color_3: str
    color_4: str


@dataclass
class Team:
    info: TeamInfo
    stats: TeamStats


@dataclass
class Matchup:
    """A game between two teams."""

    away: Team
    home: Team
    div: bool


class DataLoader:
    teams: pd.DataFrame
    pbp_data_train: pd.DataFrame
    pbp_data_test: pd.DataFrame

    def __init__(self):
        self.__load_teams()
        self.__load_raw_pbp_data()

    def __load_raw_pbp_data(self) -> None:
        """Load play-by-play data from nflfastR."""

        current_year = datetime.now().year

        with Timer("PBP loading"):
            # all seasons of pbp data from 1999 - current_year
            self.pbp_data_train = pd.concat(
                [
                    pd.read_parquet(_PBP_URL.format(year=y))
                    for y in range(1999, current_year - 1)
                ],
                ignore_index=True,
            )

            # latest season
            self.pbp_data_test = pd.read_parquet(_PBP_URL.format(year=current_year - 1))

    def __load_teams(self) -> None:
        """Load team colors, logos, and metadata from nflverse."""

        with Timer("Teams loading"):
            self.teams = pd.read_parquet(_TEAMS_URL)
