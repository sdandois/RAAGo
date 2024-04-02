import trueskillthroughtime as ttt
import pandas as pd
from datetime import datetime, time

import math


EPSILON = 0.01
ITERATIONS = 16 # 16
BETA = 1.8  # relation between performance and winning probability (spread)
MU = 0
SIGMA = 1.125 * BETA # standard deviations of priors
GAMMA = 0.025 * BETA # time uncertainty
HANDI_SIGMA = 1 * SIGMA
HANDI1_MU = MU

MU_0 = -5.8337 # After model is applied, use this displacement




def calculate_ttt_ratings(df):
    """Calculates TrueSkillTroughTime ratings"""

    df["winner_id"] = df.apply(
        lambda row: (
            str(row["black_player_id"])
            if row["result"] == "black"
            else str(row["white_player_id"])
        ),
        axis=1,
    )
    df["looser_id"] = df.apply(
        lambda row: (
            str(row["white_player_id"])
            if row["result"] == "black"
            else str(row["black_player_id"])
        ),
        axis=1,
    )
    df["winner_handicap"] = df.apply(
        lambda row: (
            "handi_" + str(row["handicap"]) if row["result"] == "black" else "handi_0"
        ),
        axis=1,
    )
    df["looser_handicap"] = df.apply(
        lambda row: (
            "handi_0" if row["result"] == "black" else "handi_" + str(row["handicap"])
        ),
        axis=1,
    )

    def winner_team(row):
        if row["result"] == "black":
            return [str(row["black_player_id"]), "handi_1", "handi_0"]
        else:
            return [str(row["white_player_id"])]

    def looser_team(row):
        if row["result"] == "white":
            return [str(row["black_player_id"]), "handi_1", "handi_0"]
        else:
            return [str(row["white_player_id"])]

    def winner_team_weights(row):
        handicap = float(row.handicap)
        dividend = 2.0 + handicap
        if row["result"] == "black":
            return [1.0 / dividend, handicap, 1.0 / dividend]
        else:
            return [1.0]

    def looser_team_weights(row):
        handicap = float(row.handicap)
        dividend = 2.0 + handicap
        if row["result"] == "white":
            return [1.0 / dividend, handicap / dividend, 1.0 / dividend]
        else:
            return [1.0]

    df["winner_team"] = df.apply(winner_team, axis=1)
    df["looser_team"] = df.apply(looser_team, axis=1)
    df["winner_team_weights"] = df.apply(winner_team_weights, axis=1)
    df["looser_team_weights"] = df.apply(looser_team_weights, axis=1)

    columns = zip(df.winner_team, df.looser_team)
    composition = [[w, l] for w, l in columns]

    weights = [
        [ww, lw] for ww, lw in zip(df.winner_team_weights, df.looser_team_weights)
    ]
    times = [date_to_num(t) for t in df.date]

    priors_handi = {
        "handi_1": ttt.Player(ttt.Gaussian(HANDI1_MU, HANDI_SIGMA), beta=0, gamma=0),
        "handi_0": ttt.Player(ttt.Gaussian(MU, HANDI_SIGMA), beta=0, gamma=0),
    }

    h = ttt.History(
        composition=composition,
        times=times,
        priors=priors_handi,
        mu=MU,
        sigma=SIGMA,
        gamma=GAMMA,
        weights=weights,
    )
    h.convergence(epsilon=EPSILON, iterations=ITERATIONS)  # 16

    learning_curves = h.learning_curves()

    # Create ratings df from learning_curves
    l = []

    for entry in learning_curves.items():
        for tup in entry[1]:
            l.append([entry[0], tup[0], tup[1].mu, tup[1].sigma])

    new_ratings = pd.DataFrame(l, columns=["player_id", "time", "mu", "sigma"])

    new_ratings = new_ratings[~new_ratings["player_id"].isin(["handi_0", "handi_1"])]
    new_ratings["player_id"] = pd.to_numeric(new_ratings["player_id"])
    new_ratings["date"] = new_ratings["time"].apply(num_to_date)

    new_ratings["mu"] = new_ratings["mu"].apply(lambda x: x + MU_0)
    new_ratings["mu"] = new_ratings["mu"].apply(converted_mu)

    le = h.log_evidence()
    mean_evidence = math.exp(le / h.size)

    return new_ratings, le, mean_evidence


def date_to_num(date):
    return datetime.combine(date, time.min).timestamp() / (60 * 60 * 24)


def num_to_date(num):
    return datetime.fromtimestamp(num * (60 * 60 * 24)).date()


def converted_mu(x):
    val = x - (x < 0) + (x > 0)

    return val
