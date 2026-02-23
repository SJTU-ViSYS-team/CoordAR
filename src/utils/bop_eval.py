import json
import logging
import os
import pickle
import subprocess
from tqdm import tqdm

try:
    import mmcv
    import mmcv.load
except ImportError:
    import mmengine as mmcv
import numpy as np
import os.path as osp
from tabulate import tabulate


from src.third_party.bop_toolkit.bop_toolkit_lib.misc import get_error_signature
from src.utils.system import increment_path
from src.utils.status import Status
from src.third_party.bop_toolkit.bop_toolkit_lib import inout

logger = logging.getLogger(__name__)


def save_test_predictions(predictions, out_dir):
    results = []
    for pred in predictions:
        TCO = pred["TCO"]
        result = dict(
            scene_id=f'{pred["scene_id"]}',
            im_id=f'{pred["im_id"]}',
            gt_id=f'{pred["gt_id"]}',
            obj_id=f'{pred["obj_id"]}',
            bbox=" ".join([f"{b:.4f}" for b in pred["bbox"]]),
            mssd_recall=f"{pred['mssd_recall']:.3f}",
            add_recall=f"{pred['add_recall']:.3f}",
            TCO=" ".join([f"{r:.4f}" for r in TCO.reshape(-1).tolist()]),
            TCO_gt=" ".join([f"{r:.4f}" for r in pred["TCO_gt"].reshape(-1).tolist()]),
            TCO_ref=" ".join(
                [f"{r:.4f}" for r in pred["TCO_ref"].reshape(-1).tolist()]
            ),
        )
        results.append(result)
    pred_path = f"{out_dir}/predictions.json"
    inout.save_json(pred_path, results)

    keys = ["scene_id", "im_id", "gt_id", "obj_id", "mssd_recall", "add_recall"]
    pred_path = f"{out_dir}/metrics.csv"
    with open(pred_path, "w") as fp:
        fp.write(f"{','.join(keys)}\n")
        for result in results:
            line_items = []
            for k in keys:
                line_items.append(f"{result[k]}")
            fp.write(f"{','.join(line_items)}\n")
    print("predictions saved to ", pred_path)


def save_and_eval_results(
    dp_split,
    eval_dir,
    pred_name,
    predictions,
    targets_filename,
    error_types,
    ntop=-1,
    obj_ids=None,
):
    results = []
    masks = []
    additional_col = {}
    any_success = False
    for pred in predictions:
        if pred["status"] == Status.SUCCESS:
            any_success = True
            TCO = pred["TCO"]
            result = dict(
                scene_id=f'{pred["scene_id"]}',
                im_id=f'{pred["im_id"]}',
                obj_id=f'{pred["obj_id"]}',
                score=f'{pred["score"]:.4f}',
                R=" ".join([f"{r:.4f}" for r in TCO[:3, :3].reshape(-1).tolist()]),
                t=" ".join([f"{tt:.4f}" for tt in TCO[:3, 3].reshape(-1).tolist()]),
                time="-1",
            )
            results.append(result)
            additional_col.setdefault("det_id", []).append(pred["det_id"])
            if "vis_mask_rle" in pred:
                masks.append(
                    dict(
                        vis_mask_rle=pred["vis_mask_rle"],
                        full_mask_rle=pred["full_mask_rle"],
                        vis_mask_full_res_rle=pred["vis_mask_full_res_rle"],
                    )
                )
    if not any_success:
        print("Warning! all prediction Failed.")
        return
    keys = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
    eval_dir = increment_path(eval_dir, exist_ok=False, mkdir=True)
    pred_path = f"{eval_dir}/{pred_name}"
    with open(pred_path, "w") as fp:
        fp.write(f"{','.join(keys)}\n")
        for result in results:
            line_items = []
            for k in keys:
                line_items.append(f"{result[k]}")
            fp.write(f"{','.join(line_items)}\n")
    print("predictions saved to ", pred_path)

    mask_path = f"{eval_dir}/{pred_name.replace('.csv', '_mask.json')}"
    inout.save_json(mask_path, masks)
    print("mask saved to ", mask_path)

    # save for refinement
    refinement_file_path = f"{eval_dir}/{pred_name.replace('.csv', '.pkl')}"
    save_for_refinement(refinement_file_path, predictions)

    # additional col
    additional_col_path = (
        f"{eval_dir}/{pred_name.replace('.csv', '_additional_col.json')}"
    )
    inout.save_json(additional_col_path, additional_col)

    eval_results(
        dp_split, eval_dir, pred_path, targets_filename, error_types, ntop, obj_ids
    )


def save_for_refinement(filename, predictions):
    results = {}
    for pred in predictions:
        if pred["status"] == Status.SUCCESS:
            key = f"{pred['scene_id']}/{pred['im_id']}"
            im_results = {}
            im_results["R"] = pred["TCO"][:3, :3]
            im_results["t"] = pred["TCO"][:3, 3] * 0.001
            im_results["score"] = pred["score"]
            im_results["bbox_est"] = pred["bbox"]  # xywh
            im_results["obj_id"] = pred["obj_id"]
            if "vis_mask_full_res_rle" in pred:
                im_results["mask"] = pred["vis_mask_full_res_rle"]
            results.setdefault(key, []).append(im_results)
    return pickle.dump(results, open(filename, "wb"))


def save_cmd_to_file(cmd, filename):
    """Save the command to a file."""
    with open(filename, "w") as f:
        f.write(" ".join(cmd) + "\n")


def eval_results(
    dp_split, eval_dir, result_path, targets_filename, error_types, ntop, obj_ids=None
):

    result_dir = os.path.dirname(result_path)
    result_names = [os.path.basename(result_path)]

    # used cached eval results if exists
    result_names_str = ",".join(result_names)
    bop_path = f"{dp_split['base_path']}/.."
    eval_cmd = [
        "python",
        "src/third_party/bop_toolkit/scripts/eval_bop19_pose.py",
        "--datasets_path={}".format(bop_path),
        "--results_path={}".format(result_dir),
        "--result_filenames={}".format(result_names_str),
        "--eval_path={}".format(eval_dir),
        "--targets_filename={}".format(targets_filename),
        "--visib_gt_min=-1",
        "--errors",
        *error_types,
    ]
    save_cmd_to_file(eval_cmd, osp.join(eval_dir, "eval_cmd.txt"))

    eval_env = os.environ.copy()
    eval_env.update({"BOP_PATH": bop_path})
    if subprocess.call(eval_cmd, env=eval_env) != 0:
        raise RuntimeError("evaluation failed.")

    load_and_print_val_scores_tab(
        ntop,
        dp_split,
        eval_root=eval_dir,
        result_names=result_names,
        error_types=error_types,
        obj_ids=obj_ids,
        targets_filename=targets_filename,
    )


def is_weighted_average_metric(error_type):
    if error_type in ["mspd", "mssd", "vsd"]:
        return True
    return False


def get_object_nums_from_targets(targets_path):
    """stat the number of each object given a targets json file in BOP
    format."""
    assert osp.exists(targets_path), targets_path
    targets = mmcv.load(targets_path)

    obj_nums_dict = {}
    for target in targets:
        obj_id = target["obj_id"]
        if obj_id not in obj_nums_dict:
            obj_nums_dict[obj_id] = 0
        obj_nums_dict[obj_id] += target["inst_count"]
    res_obj_nums_dict = {
        str(key): obj_nums_dict[key] for key in sorted(obj_nums_dict.keys())
    }
    return res_obj_nums_dict


def redirect_logger_to_file(logger, log_file_path):
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_and_print_val_scores_tab(
    ntop,
    dp_split,
    eval_root,
    result_names,
    error_types=["projS", "ad", "reteS"],
    obj_ids=None,
    print_all_objs=False,
    targets_filename="test_targets_bop19.json",
):

    vsd_deltas = {
        "hb": 15,
        "hbs": 15,
        "icbin": 15,
        "icmi": 15,
        "itodd": 5,
        "lm": 15,
        "lmo": 15,
        "ruapc": 15,
        "tless": 15,
        "tudl": 15,
        "tyol": 15,
        "ycbv": 15,
        "ycbvposecnn": 15,
        "robi": 15,
        "real275": 15,
    }
    vsd_delta = vsd_deltas.get(dp_split["name"].split("_")[0])

    if any(is_weighted_average_metric(err_type) for err_type in error_types):
        obj_nums_dict = get_object_nums_from_targets(
            osp.join(dp_split["base_path"], targets_filename)
        )
    else:
        obj_nums_dict = None

    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    # visib_gt_min = 0.1

    redirect_logger_to_file(logger, osp.join(eval_root, "eval.log"))

    for result_name in tqdm(result_names):
        logger.info(
            "====================================================================="
        )
        big_tab_row = []
        for error_type in error_types:
            result_name = result_name.replace(".csv", "")
            # logger.info(f"************{result_name} *** [{error_type}]*******************")
            if error_type == "vsd":
                error_signs = [
                    get_error_signature(
                        error_type, ntop, vsd_delta=vsd_delta, vsd_tau=vsd_tau
                    )
                    for vsd_tau in vsd_taus
                ]
            else:
                error_signs = [get_error_signature(error_type, ntop)]
            score_roots = [
                osp.join(eval_root, result_name, error_sign)
                for error_sign in error_signs
            ]

            for score_root in score_roots:
                if osp.exists(score_root):
                    # get all score json files for this metric under this threshold
                    score_paths = {
                        osp.join(score_root, fn.name): None
                        for fn in os.scandir(score_root)
                        if ".json" in fn.name and "scores" in fn.name
                    }

                    tab_obj_col = summary_scores(
                        dp_split,
                        score_paths,
                        error_type,
                        print_all_objs=print_all_objs,
                        obj_ids=obj_ids,
                        obj_nums_dict=obj_nums_dict,
                    )
                    # print single metric with obj in col here
                    logger.info(f"************{result_name} *********************")
                    tab_obj_col_log_str = tabulate(
                        tab_obj_col,
                        tablefmt="plain",
                        # floatfmt=floatfmt
                    )
                    logger.info("\n{}".format(tab_obj_col_log_str))
                    #####
                    big_tab_row.append(tab_obj_col.T)  # objs in row

                else:
                    logger.warning("{} does not exist.".format(score_root))
                    raise RuntimeError("{} does not exist.".format(score_root))

        if len(big_tab_row) > 0:
            # row: obj in row
            # col: obj in col
            logger.info(f"************{result_name} *********************")
            if len(big_tab_row) == 1:
                res_log_tab = big_tab_row[0]
            else:
                res_log_tab = np.concatenate(
                    [big_tab_row[0]] + [_tab[1:, :] for _tab in big_tab_row[1:]],
                    axis=0,
                )

            new_res_log_tab = maybe_average_vsd_scores(res_log_tab)
            new_res_log_tab_col = new_res_log_tab.T

            if len(new_res_log_tab) < len(
                new_res_log_tab_col
            ):  # print the table with more rows later
                log_tabs = [new_res_log_tab, new_res_log_tab_col]
                suffixes = ["row", "col"]
            else:
                log_tabs = [new_res_log_tab_col, new_res_log_tab]
                suffixes = ["col", "row"]
            for log_tab_i, suffix in zip(log_tabs, suffixes):
                dump_tab_name = osp.join(
                    eval_root, f"{result_name}_tab_obj_{suffix}.txt"
                )
                log_tab_i_str = tabulate(
                    log_tab_i,
                    tablefmt="plain",
                    # floatfmt=floatfmt
                )
                logger.info("\n{}".format(log_tab_i_str))
                with open(dump_tab_name, "w") as f:
                    f.write("{}\n".format(log_tab_i_str))
    logger.info("{}".format(eval_root))


def is_auc_metric(error_type):
    if error_type in ["AUCadd", "AUCadi", "AUCad", "vsd", "mssd", "mspd", "re", "te"]:
        return True
    return False


def get_thr(score_path):
    # used for sorting score json files
    # scores_th:2.000_min-visib:0.100.json
    # rete: scores_th:10.000-10.000_min-visib:-1.000.json
    # NOTE: assume the same threshold (currently can deal with rete, rete_s)
    return float(
        score_path.split("/")[-1].replace("scores_th=", "").split("_")[0].split("-")[0]
    )


def simplify_float_str(float_str):
    value = float(float_str)
    if value == int(value):
        return str(int(value))
    return float_str


def get_thr_str(score_path):
    # path/to/scores_th=2.000_min-visib:0.100.json  --> 2
    # rete: path/to/scores_th=10.000-10.000_min-visib:-1.000.json --> 10
    thr_str = score_path.split("/")[-1].split("_")[1]
    thr_str = thr_str.split("=")[-1]
    if "-" in thr_str:
        thr_str_split = thr_str.split("-")
        simple_str_list = [simplify_float_str(_thr) for _thr in thr_str_split]
        if len(set(simple_str_list)) == 1:
            res_thr_str = simple_str_list[0]
        else:
            res_thr_str = "-".join(simple_str_list)
    else:
        res_thr_str = simplify_float_str(thr_str)
    return res_thr_str


def summary_scores(
    dp_split,
    score_paths,
    error_type,
    print_all_objs=False,
    obj_ids=None,
    obj_nums_dict=None,
):

    sorted_score_paths = sorted(score_paths.keys(), key=get_thr)

    min_max_thr_str = None
    obj_recalls_dict = {}
    if is_auc_metric(error_type):
        min_thr_str = get_thr_str(sorted_score_paths[0])
        max_thr_str = get_thr_str(sorted_score_paths[-1])
        min_max_thr_str = f"{min_thr_str}:{max_thr_str}"

    tabs_col2 = []
    for score_path in sorted_score_paths:
        score_dict = mmcv.load(score_path)
        if obj_ids is None:
            sel_obj_ids = [int(_id) for _id in score_dict["obj_recalls"].keys()]
        else:
            sel_obj_ids = obj_ids

        thr_str = get_thr_str(score_path)
        # logging the results with tabulate
        # tab_header = ["objects", "{}[{}](%)".format(error_type, thr_str)]
        tab_header = [
            "objects",
            "{}_{}".format(error_type, thr_str),
        ]  # 2 columns, objs in col
        cur_tab_col2 = [tab_header]
        for _id, _recall in score_dict["obj_recalls"].items():
            obj_name = str(_id)
            if int(_id) in sel_obj_ids:
                cur_tab_col2.append([obj_name, f"{_recall * 100:.2f}"])
                if min_max_thr_str is not None:  # for AUC metrics
                    if obj_name not in obj_recalls_dict:
                        obj_recalls_dict[obj_name] = []
                    obj_recalls_dict[obj_name].append(_recall)
            else:
                if print_all_objs:
                    cur_tab_col2.append([obj_name, "-"])

        # mean of selected objs
        num_objs = len(sel_obj_ids)
        if num_objs > 1:
            sel_obj_recalls = [
                _recall
                for _id, _recall in score_dict["obj_recalls"].items()
                if int(_id) in sel_obj_ids
            ]
            if not is_weighted_average_metric(error_type):
                mean_obj_recall = np.mean(sel_obj_recalls)
            else:
                assert obj_nums_dict is not None
                sel_obj_nums = np.array(
                    [_v for _k, _v in obj_nums_dict.items() if int(_k) in sel_obj_ids]
                )
                sel_obj_weights = sel_obj_nums / sum(sel_obj_nums)
                mean_obj_recall = sum(sel_obj_weights * np.array(sel_obj_recalls))
            cur_tab_col2.append(
                ["Avg({})".format(num_objs), f"{mean_obj_recall * 100:.2f}"]
            )

        cur_tab_col2 = np.array(cur_tab_col2)
        tabs_col2.append(cur_tab_col2)

    if len(tabs_col2) == 1:
        return tabs_col2[0]
    else:
        if min_max_thr_str is None:  # not AUC metrics, concat
            res_tab = np.concatenate(
                [tabs_col2[0]] + [_tab[:, 1:2] for _tab in tabs_col2[1:]],
                axis=1,
            )
        else:  # AUC metrics, mean
            auc_header = [
                "objects",
                "{}_{}".format(error_type, min_max_thr_str),
            ]  # 2 columns, objs in col
            res_tab = [auc_header]
            obj_aucs = []
            obj_nums = []
            for obj_name in tabs_col2[0][1:-1, 0].tolist():
                if obj_name in obj_recalls_dict:
                    cur_auc = np.mean(obj_recalls_dict[obj_name])
                    obj_aucs.append(cur_auc)
                    if obj_nums_dict is not None:
                        obj_nums.append(obj_nums_dict[str(_id)])
                    res_tab.append([obj_name, f"{cur_auc * 100:.2f}"])
            if is_weighted_average_metric(error_type):
                assert len(obj_nums) == len(
                    obj_aucs
                ), f"{len(obj_nums)} != {len(obj_aucs)}"
                obj_weights = np.array(obj_nums) / sum(obj_nums)
                mean_obj_auc = sum(np.array(obj_aucs) * obj_weights)
            else:
                mean_obj_auc = np.mean(obj_aucs)
            res_tab.append(
                ["Avg({})".format(len(obj_aucs)), f"{mean_obj_auc * 100:.2f}"]
            )
            res_tab = np.array(res_tab)
        return res_tab


def maybe_average_vsd_scores(res_log_tab):
    # obj in row, scores in col
    if "vsd_0.050:0.500" in res_log_tab[:, 0]:
        vsd_rows = [
            _r
            for _r in range(res_log_tab.shape[0])
            if res_log_tab[_r, 0] == "vsd_0.050:0.500"
        ]
        vsd_mean = np.mean(res_log_tab[vsd_rows, 1:].astype("float32"), 0)
        vsd_mean_row = np.array(
            ["vsd_0.050:0.500"] + [f"{_v:.2f}" for _v in vsd_mean],
            dtype=res_log_tab.dtype,
        )
        new_res_log_tab = []
        vsd_cnt = 0
        for row_i, log_row in enumerate(res_log_tab):
            if row_i not in vsd_rows:
                new_res_log_tab.append(log_row)
            else:
                if vsd_cnt == 0:
                    new_res_log_tab.append(vsd_mean_row)
                vsd_cnt += 1
        new_res_log_tab = np.array(new_res_log_tab)
    else:
        new_res_log_tab = res_log_tab
    return new_res_log_tab
