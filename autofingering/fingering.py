import pandas as pd
import numpy as np
from collections import Counter


def pitch_to_key(pitch: str):
    posx = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}[pitch[0]]
    posy = 0

    if pitch[1].isdigit():
        posx += (int(pitch[1]) - 4) * 7
    elif pitch[1] == "#":
        posy = 1
        posx += (int(pitch[2]) - 4) * 7
    elif pitch[1] == "b" or pitch[1] == "-":
        posy = 1
        posx += (int(pitch[2]) - 4) * 7 - 1

    return (posx, posy)


def note_to_diff(fingering_data, limit=15):
    pos_x, pos_y = zip(*fingering_data.pitch.map(pitch_to_key))
    series_x = pd.Series(pos_x)
    series_y = pd.Series(pos_y)
    diffs = list(
        zip(
            series_x.diff()
            .fillna(0, downcast="infer")
            .apply(lambda x: limit if x > limit else x)
            .apply(lambda x: -limit if x < -limit else x),
            series_y.diff().fillna(0, downcast="infer"),
        )
    )
    return diffs


def count_fingering(fingering_data, limit=15):
    hidden_state = list(
        zip(
            fingering_data.fingernum.shift(fill_value=0),
            fingering_data.fingernum,
        )
    )

    pos_x, pos_y = zip(*fingering_data.pitch.map(pitch_to_key))

    model = pd.DataFrame(
        {"hidden_state": hidden_state, "pos_x": pos_x, "pos_y": pos_y}
    )

    model["pos_diff"] = list(
        zip(
            model.pos_x.diff()
            .fillna(0, downcast="infer")
            .apply(lambda x: limit if x > limit else x)
            .apply(lambda x: -limit if x < -limit else x),
            model.pos_y.diff().fillna(0, downcast="infer"),
        )
    )

    # First observation only
    init = Counter([model.hidden_state[0][1]])

    # Without first observation
    transition = Counter(model.hidden_state[1:])

    # Emission
    emission = {
        state: Counter(model[model.hidden_state == state].pos_diff)
        for state in set(model.hidden_state[1:])
    }

    return (init, transition, Counter(emission))


def normalize(v):
    return v / v.sum(axis=0)


def init_count_to_prob(init_count):
    init_prob = np.zeros(5)
    for key, value in init_count.items():
        if key < 0:
            init_prob[-key - 1] = value
        else:
            init_prob[key - 1] = value
    return normalize(init_prob)


def transition_count_to_prob(transition_count):
    transition_prob = np.zeros((5, 5))
    for key, value in transition_count.items():
        if key[0] < 0 and key[1] < 0:
            transition_prob[-key[0] - 1, -key[1] - 1] = value
        else:
            transition_prob[key[0] - 1, key[1] - 1] = value

    return np.apply_along_axis(normalize, axis=1, arr=transition_prob)


def series_to_matrix(emission_prob):
    out_prob = np.zeros((5, 5))
    for key, value in emission_prob.items():
        if key[0] < 0 and key[1] < 0:
            out_prob[-key[0] - 1, -key[1] - 1] = value
        else:
            out_prob[key[0] - 1, key[1] - 1] = value

    return out_prob


def emission_count_to_prob(emission_count):
    prob_df = (
        pd.DataFrame.from_dict(emission_count).fillna(0, downcast="infer") + 1
    ).apply(normalize, axis=0)

    prob_dict = {
        out: series_to_matrix(prob_df.loc[out]) for out in prob_df.index
    }

    return prob_dict


def decoding(init_prob, transition, out_prob, observations, hand):
    n_state = len(init_prob)
    obs_len = len(observations)

    delta = np.zeros((n_state, obs_len + 1))
    psi = np.zeros((n_state, obs_len), dtype=int)
    delta[:, 0] = np.log(init_prob)

    for i, (pitch, time) in enumerate(
        zip(observations.pitch_diff, observations.time_diff)
    ):
        delta_mat = np.tile(delta[:, i], (n_state, 1)).transpose()
        prod = delta_mat + np.log(transition) + np.log(out_prob[pitch])
        if time < 0.03:
            if hand == "R":
                if pitch[0] > 0:
                    prod[np.tril_indices(n_state)] -= 5
                else:
                    prod[np.triu_indices(n_state)] -= 5
            else:
                if pitch[0] > 0:
                    prod[np.triu_indices(n_state)] -= 5
                else:
                    prod[np.tril_indices(n_state)] -= 5

        delta[:, i + 1] = np.amax(prod, axis=0)
        psi[:, i] = prod.argmax(axis=0) + 1

    opt_path = [np.argmax(delta[:, obs_len]) + 1]

    for i in range(obs_len - 1, -1, -1):
        opt_path.append(psi[opt_path[-1] - 1, i])

    return opt_path[::-1]
