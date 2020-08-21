import os
import music21
import pandas as pd
import numpy as np


class Sheet:
    def __init__(self, path):
        self.path = path
        self.sheet = music21.converter.parse(path)

    def select_part(self, part_num):
        if hasattr(self.sheet, "parts"):
            return self.sheet.parts[part_num].flat
        else:
            return self.sheet.flat

    def build_note_info(self, part_num):
        notes = []
        part = self.select_part(part_num)

        for idx, note in enumerate(part):
            if note.duration.quarterLength == 0:
                continue

            if note.tie and (
                note.tie.type == "continue" or note.tie.type == "stop"
            ):
                continue

            if note.isNote:
                notes.append(
                    {
                        "id": idx,
                        "pitch": note.nameWithOctave,
                        "time": note.offset,
                    }
                )

            if note.isChord:
                for n in note.notes:
                    notes.append(
                        {
                            "id": idx,
                            "pitch": n.nameWithOctave,
                            "time": note.offset,
                        }
                    )

        return pd.DataFrame(notes)

    def add_fingernum(self, part_num, finger_info):
        part = self.select_part(part_num)

        for idx, note in enumerate(part):
            if note.duration.quarterLength == 0:
                continue

            if note.tie and (
                note.tie.type == "continue" or note.tie.type == "stop"
            ):
                continue

            finger_num = list(finger_info[finger_info.id == idx].finger_num)

            if note.isNote:
                f = music21.articulations.Fingering(finger_num[0])
                note.articulations.append(f)
            if note.isChord:
                for num in finger_num:
                    f = music21.articulations.Fingering(num)
                    note.articulations.append(f)

    def write_on(self, path):
        self.sheet.write("xml", fp=path)
