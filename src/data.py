from datetime import datetime

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from src.timer import Timer


def load_raw_pbp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load play-by-play data from nflfastR.

    Returns a tuple of (training_data, testing_data) where:
    - training_data: all seasons through last year
    - testing_data: the current year's season
    """

    current_year = datetime.now().year

    ro.r("library(nflfastR)")
    ro.r("options(nflreadr.verbose = FALSE)")

    with Timer("PBP loading"):
        with localconverter(ro.default_converter + pandas2ri.converter):
            training_data = ro.r(f"nflfastR::load_pbp(seasons={1999}:{current_year-2})")
            testing_data = ro.r(f"nflfastR::load_pbp(seasons={current_year-1})")

    return training_data, testing_data
