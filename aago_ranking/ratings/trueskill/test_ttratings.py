from unittest import TestCase

import pandas as pd
import numpy as np
import os

from . import tttratings


class TTRatingsTest(TestCase):
    """Tests de tttratings"""

    def setUp(self) -> None:
        self.games = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "games_dump.csv"),
        )
        
        self.games['date'] = pd.to_datetime(self.games['date'])

    def test_calculate_ratings(self):
        ratings = tttratings.calculate_ttt_ratings(self.games)

        pass
