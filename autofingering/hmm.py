import pandas as pd
import numpy as np


def viterbi(init_prob, transition, out_prob, observations):
    n_state = len(init_prob)
    obs_len = len(observations)

    delta = init_prob

    print(
        [
            delta[i] * transition[i, 1] * out_prob[(i+1, 1)][observations[0]]
            for i in range(0, n_state)
        ]
    )
