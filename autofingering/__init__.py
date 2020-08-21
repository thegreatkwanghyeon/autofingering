import os
import pandas as pd
import numpy as np
import fingering
import sheetreader
from collections import Counter, defaultdict

pig_format = [
    "id",
    "onset",
    "offset",
    "pitch",
    "onsetvel",
    "offsetvel",
    "hand",
    "fingernum",
]

file_list = os.listdir("data")

right_init = Counter()
right_transition_count = Counter()
right_emission = defaultdict(Counter)

left_init = Counter()
left_transition_count = Counter()
left_emission = defaultdict(Counter)

leap_limit = 15

for idx, file in enumerate(file_list):
    path = "data/" + file
    data_size = len(file_list)

    print(f"Processing: {path} ({idx}/{data_size})")

    data = pd.read_csv(path, sep="\t", header=0, names=pig_format)

    if data.fingernum.dtype == object:
        data.fingernum = data.fingernum.apply(
            lambda x: x.split("_")[0]
        ).astype("int")

    left_hand = data[data.fingernum < 0]
    right_hand = data[data.fingernum > 0]

    init, transition, emission = fingering.count_fingering(
        right_hand, limit=leap_limit
    )
    right_init += init
    right_transition_count += transition
    for k, counter in emission.items():
        right_emission[k].update(counter)

    init, transition, emission = fingering.count_fingering(
        left_hand, limit=leap_limit
    )
    left_init += init
    left_transition_count += transition
    for k, counter in emission.items():
        left_emission[k].update(counter)


right_transition_prob = fingering.transition_count_to_prob(
    right_transition_count
)
left_transition_prob = fingering.transition_count_to_prob(
    left_transition_count
)

right_init_prob = fingering.init_count_to_prob(right_init)
left_init_prob = fingering.init_count_to_prob(left_init)


right_output_prob = fingering.emission_count_to_prob(right_emission)
left_output_prob = fingering.emission_count_to_prob(left_emission)

sheet = sheetreader.Sheet("sheet/etude.mxl")

right_part = sheet.build_note_info(0)
left_part = sheet.build_note_info(1)

right_part["time_diff"] = right_part.time.diff()
left_part["time_diff"] = left_part.time.diff()
right_part["pitch_diff"] = fingering.note_to_diff(right_part)
left_part["pitch_diff"] = fingering.note_to_diff(left_part)

right_num = fingering.decoding(
    right_init_prob, right_transition_prob, right_output_prob, right_part, "R"
)


left_num = fingering.decoding(
    left_init_prob, left_transition_prob, left_output_prob, left_part, "L"
)

right_part["finger_num"] = right_num[1:]
left_part["finger_num"] = left_num[1:]

sheet.add_fingernum(0, right_part)
sheet.add_fingernum(1, left_part)

sheet.write_on("sheet/etude_out.mxl")
