import argparse

import pandas as pd

from bop_toolkit_lib import inout

# linemod object id to name mapping
LM_ID2Name = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

exclude_objs = ["bowl", "cup"]


# python -m src.scripts.lm.report_lm_add --score_path "logs/predict/runs/2026-02-23_19-39-07/eval/ADD(-S)/cir-lm-test/error=ad_ntop=-1/scores_th=0.100_min-visib=-1.000.json"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report LM Add")
    parser.add_argument("--score_path", type=str, help="Path to the score file")

    args = parser.parse_args()

    data = inout.load_json(args.score_path, keys_to_int=True)
    obj_recalls = data["obj_recalls"]

    print(data)
    recalls = []
    for obj_id, obj_recall in obj_recalls.items():
        if LM_ID2Name[obj_id] in exclude_objs:
            continue
        recalls.append((LM_ID2Name[obj_id], obj_recall))

    df = pd.DataFrame(recalls, columns=["Object", "Recall"])
    mean_recall = df["Recall"].mean()

    print(df)
    print(f"Mean Recall: {mean_recall:.3f}")

    print("latex code: ", " & ".join(df["Recall"].apply(lambda x: f"{x*100:.1f}")))
