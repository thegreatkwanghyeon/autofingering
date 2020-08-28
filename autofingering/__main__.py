import os
import argparse
import pandas as pd
import numpy as np
import fingering
import hand
import sheetreader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatic annotation of piano fingering based on HMM."
    )

    parser.add_argument(
        "input_sheet",
        type=str,
        help="Input music XML file name."
    )

    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        default="output.mxl",
        help="Annotated output music XML file name."
    )

    parser.add_argument(
        "-i",
        "--in-params",
        type=str,
        default="",
        help="File name of pre-trained hand model params."
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="",
        help="PIG data directory path.")

    parser.add_argument(
        "-p",
        "--out-params",
        type=str,
        default="",
        help="File name of model parameters. It is only necessary to save the learned parameters."
    )

    parser.add_argument(
        "--right-beam",
        type=int,
        default=0,
        help="Specify right hand beam number."
    )

    parser.add_argument(
        "--left-beam",
        type=int,
        default=1,
        help="Specify left hand beam number."
    )

    args = parser.parse_args()

    sheet = sheetreader.Sheet(args.input_sheet)

    left_hand = hand.Hand("L")
    right_hand = hand.Hand("R")

    # Train the model from data
    if args.data != "":
        directory = os.listdir(args.data)

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

        left_notes = list()
        right_notes = list()
        for file in directory:
            path = args.data + "/" + file
            data = pd.read_csv(path, sep="\t", header=0, names=pig_format)

            if data.fingernum.dtype == object:
                data.fingernum = data.fingernum.apply(
                    lambda x: x.split("_")[0]
                ).astype("int")

            left_notes.append(data[data.fingernum < 0])
            right_notes.append(data[data.fingernum > 0])

        left_hand.build_from_data(left_notes)
        right_hand.build_from_data(right_notes)

    elif args.in_params != "":
        param = np.load(args.in_params, allow_pickle=True)
        left_hand.build_from_params(
            init=param["left_init"],
            transition=param["left_transition"],
            emission=param["left_emission"].item()
        )
        right_hand.build_from_params(
            init=param["right_init"],
            transition=param["right_transition"],
            emission=param["right_emission"].item()
        )

    else:
        raise Exception(
            """You must give PIG data directory (-d option)
                or pre-trained parameter (-i option)."""
        )

    # Write parameters in a file
    if args.out_params != "":
        np.savez(args.out_params,
                 left_init=left_hand.init_prob,
                 left_transition=left_hand.transition_prob,
                 left_emission=left_hand.emission_prob,
                 right_init=right_hand.init_prob,
                 right_transition=right_hand.transition_prob,
                 right_emission=right_hand.emission_prob
                 )

    right_part = sheet.build_note_info(args.right_beam)
    left_part = sheet.build_note_info(args.left_beam)

    right_part["time_diff"] = right_part.time.diff()
    left_part["time_diff"] = left_part.time.diff()
    right_part["pitch_diff"] = fingering.note_to_diff(right_part)
    left_part["pitch_diff"] = fingering.note_to_diff(left_part)

    right_num = right_hand.decoding(right_part)
    left_num = left_hand.decoding(left_part)

    right_part["finger_num"] = right_num[1:]
    left_part["finger_num"] = left_num[1:]

    sheet.add_fingernum(args.right_beam, right_part)
    sheet.add_fingernum(args.left_beam, left_part)

    sheet.write_on(args.out_file)
