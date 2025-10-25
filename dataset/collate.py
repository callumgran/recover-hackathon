import torch
import math

def build_meta_features(batch):
    B = batch["X"].shape[0]

    dmg_zip = batch["damage_address_zip_code"].to(torch.int32)
    off_zip = batch["recover_office_zip_code"].to(torch.int32)

    dmg_p2 = (dmg_zip // 100).clamp(min=0, max=99).to(torch.float32) / 99.0
    off_p2 = (off_zip // 100).clamp(min=0, max=99).to(torch.float32) / 99.0

    same_zip = (dmg_zip == off_zip).to(torch.float32)
    zip_delta_bucket = (dmg_zip.sub(off_zip).abs().clamp(max=9999) // 50).to(torch.float32) / 200.0
    same_area_zip2 = (dmg_zip.ge(0) & off_zip.ge(0) & (dmg_zip.sub(off_zip).abs() < 100)).to(torch.float32)

    office_dist = batch["office_distance"].to(torch.float32)
    office_km_log1p = torch.log1p((office_dist / 1000.0).clamp(min=0)) / 10.0

    year  = batch["case_creation_year"].to(torch.float32)
    month = batch["case_creation_month"].to(torch.float32).clamp(min=1, max=12)
    year_rel  = (year - 2020.0) / 10.0
    is_winter = ((month == 11) | (month == 12) | (month == 1) | (month == 2)).to(torch.float32)

    scalars = torch.stack([
        dmg_p2, off_p2, same_zip, zip_delta_bucket, same_area_zip2,
        office_km_log1p, year_rel, is_winter
    ], dim=1)

    return scalars

OPERATION_FAMILIES = {
    # Demolition operations (commonly start workflow)
    "demolish":   [44, 49, 52, 5, 62, 58, 63, 48, 314, 56, 160],
    # Installation/New operations (commonly follow demolish)
    "install":    [46, 70, 53, 61, 11, 74, 315, 65, 162],
    # Remount/Repair operations
    "repair":     [156, 166, 173, 175, 177, 185, 165],
    # Cleaning/Finishing operations (commonly at end)
    "finishing":  [104, 108, 136, 257, 256, 204],
    # Kitchen-specific operations
    "kitchen":    [151, 153, 161, 164, 168, 170, 182],
    # Structural operations
    "structural": [55, 313, 386],
}

DEP_FEATURE_ORDER = [
    "num_ops_visible",            # 0
    "multi_ops_flag",             # 1
    "demolish_ops_visible",       # 2
    "install_ops_visible",        # 3
    "repair_ops_visible",         # 4
    "finishing_ops_visible",      # 5
    "kitchen_ops_visible",        # 6
    "structural_ops_visible",     # 7
    "demolish_completion_ratio",  # 8
    "install_completion_ratio",   # 9
    "repair_completion_ratio",    # 10
    "finishing_completion_ratio", # 11
    "kitchen_completion_ratio",   # 12
    "structural_completion_ratio",# 13
    "has_complementary_demolish", # 14
    "has_complementary_install",  # 15
    "missing_install_after_demolish", # 16
    "has_pair_44_46",             # 17
    "has_pair_49_70",             # 18
    "has_pair_52_53",             # 19
    "has_pair_62_61",             # 20
    "has_pair_314_315",           # 21
    "extensive_damage",           # 22
    "minimal_damage",             # 23
    "kitchen_complexity",         # 24
    "has_structural_work",        # 25
    "has_rebuild_phase",          # 26
]

# Default selection: DROP the brittle pair flags by default
DEFAULT_DEP_INCLUDE = {
    name: (not name.startswith("has_pair_"))  # keep everything except has_pair_* by default
    for name in DEP_FEATURE_ORDER
}

def build_dependency_features(batch: dict, num_operations: int = 388, include: dict[str, bool] | None = None) -> torch.Tensor:
    """
    Returns [B, D_dep] dependency features computed from the current room (X)
    and its context (context/context_mask). This version does NOT depend on a
    per-candidate operation; it summarizes what's visible.

    Features (order is fixed):
      0:  num_ops_visible                          (float)
      1:  multi_ops_flag                           (0/1)
      2:  demolish_ops_visible                     (count)
      3:  install_ops_visible                      (count)
      4:  repair_ops_visible                       (count)
      5:  finishing_ops_visible                    (count)
      6:  kitchen_ops_visible                      (count)
      7:  structural_ops_visible                   (count)
      8:  demolish_completion_ratio                (#demolish / |demolish set|)
      9:  install_completion_ratio                 (#install / |install set|)
      10: repair_completion_ratio                  (#repair / |repair set|)
      11: finishing_completion_ratio               (#finishing / |finishing set|)
      12: kitchen_completion_ratio                 (#kitchen / |kitchen set|)
      13: structural_completion_ratio              (#structural / |structural set|)
      14: has_complementary_demolish               (demolish_present -> 1 if install_present else 0)
      15: has_complementary_install                (install_present -> 1 if finishing_present else 0)
      16: missing_install_after_demolish           (demolish_present & ~install_present)
      17: has_pair_44_46                           (both ops present)
      18: has_pair_49_70
      19: has_pair_52_53
      20: has_pair_62_61
      21: has_pair_314_315
      22: extensive_damage                         (num_ops_visible > 5)
      23: minimal_damage                           (num_ops_visible == 1)
      24: kitchen_complexity                       (if room contains 'kjøkken' then kitchen_ops_visible else 0)
      25: has_structural_work                      (structural_ops_visible > 0)
      26: has_rebuild_phase                        (demolish_ops_visible > 0 and install_ops_visible > 0)
    """
    # Pull tensors
    X = batch["X"]                         # [B, F]
    context = batch["context"]             # [B, S, F]
    context_mask = batch["context_mask"]   # [B, S]
    rooms = batch.get("room", [""] * X.shape[0])  # list[str], optional

    B, F = X.shape
    # We assume X and context rows were built as: [ops_one_hot(=num_operations), room_one_hot(...)]
    ops_X = (X[:, :num_operations] > 0)                        # [B, O]
    ops_ctx = (context[:, :, :num_operations] > 0)             # [B, S, O]
    mask_exp = context_mask.unsqueeze(-1)                      # [B, S, 1]
    ops_ctx_any = (ops_ctx & mask_exp).any(dim=1)              # [B, O]

    # Union: ops visible either in current room or context
    ops_any = ops_X | ops_ctx_any                              # [B, O]
    num_ops_visible = ops_any.sum(dim=1).to(torch.float32)     # [B]

    # Family index tensors
    def idx_tensor(lst):
        return torch.tensor(lst, device=X.device, dtype=torch.long)

    demolish_idx   = idx_tensor(OPERATION_FAMILIES["demolish"])
    install_idx    = idx_tensor(OPERATION_FAMILIES["install"])
    repair_idx     = idx_tensor(OPERATION_FAMILIES["repair"])
    finishing_idx  = idx_tensor(OPERATION_FAMILIES["finishing"])
    kitchen_idx    = idx_tensor(OPERATION_FAMILIES["kitchen"])
    structural_idx = idx_tensor(OPERATION_FAMILIES["structural"])

    # Counts per family
    def family_count(idx):
        # Guard for empty index lists
        if idx.numel() == 0:
            return torch.zeros(B, device=X.device, dtype=torch.float32)
        return ops_any[:, idx].sum(dim=1).to(torch.float32)

    demolish_cnt   = family_count(demolish_idx)
    install_cnt    = family_count(install_idx)
    repair_cnt     = family_count(repair_idx)
    finishing_cnt  = family_count(finishing_idx)
    kitchen_cnt    = family_count(kitchen_idx)
    structural_cnt = family_count(structural_idx)

    # Ratios
    demolish_ratio   = demolish_cnt   / max(1, len(OPERATION_FAMILIES["demolish"]))
    install_ratio    = install_cnt    / max(1, len(OPERATION_FAMILIES["install"]))
    repair_ratio     = repair_cnt     / max(1, len(OPERATION_FAMILIES["repair"]))
    finishing_ratio  = finishing_cnt  / max(1, len(OPERATION_FAMILIES["finishing"]))
    kitchen_ratio    = kitchen_cnt    / max(1, len(OPERATION_FAMILIES["kitchen"]))
    structural_ratio = structural_cnt / max(1, len(OPERATION_FAMILIES["structural"]))

    # Presence flags
    demolish_present  = (demolish_cnt  > 0).to(torch.float32)
    install_present   = (install_cnt   > 0).to(torch.float32)
    finishing_present = (finishing_cnt > 0).to(torch.float32)
    structural_present= (structural_cnt> 0).to(torch.float32)

    has_compl_demolish = (install_present * demolish_present)  # if demolish seen, do we also see install?
    has_compl_install  = (finishing_present * install_present) # if install seen, do we also see finishing?
    missing_install_after_demolish = ((demolish_present == 1.0) & (install_present == 0.0)).to(torch.float32)

    # Specific pairs (presence of both ops)
    def pair_flag(a, b):
        a_idx = torch.tensor(a, device=X.device, dtype=torch.long)
        b_idx = torch.tensor(b, device=X.device, dtype=torch.long)
        a_has = ops_any[:, a_idx] if a_idx.numel() > 1 else ops_any[:, a_idx.view(1)]
        b_has = ops_any[:, b_idx] if b_idx.numel() > 1 else ops_any[:, b_idx.view(1)]
        a_has = a_has.any(dim=1) if a_has.ndim == 2 else a_has.view(-1)
        b_has = b_has.any(dim=1) if b_has.ndim == 2 else b_has.view(-1)
        return (a_has & b_has).to(torch.float32)

    has_pair_44_46   = pair_flag([44],  [46])
    has_pair_49_70   = pair_flag([49],  [70])
    has_pair_52_53   = pair_flag([52],  [53])
    has_pair_62_61   = pair_flag([62],  [61])
    has_pair_314_315 = pair_flag([314], [315])

    # Other scope/phase
    multi_ops_flag   = (num_ops_visible > 2).to(torch.float32)
    extensive_damage = (num_ops_visible > 5).to(torch.float32)
    minimal_damage   = (num_ops_visible == 1).to(torch.float32)
    has_structural_work = structural_present

    # kitchen_complexity: only if room label mentions 'kjøkken'
    if isinstance(rooms, list):
        # boolean mask per sample
        room_mask = torch.tensor([1.0 if isinstance(r, str) and ("kjøkken" in r.lower()) else 0.0 for r in rooms],
                                 device=X.device, dtype=torch.float32)
    else:
        room_mask = torch.zeros(B, device=X.device, dtype=torch.float32)
    kitchen_complexity = kitchen_cnt * room_mask

    has_rebuild_phase = ((demolish_cnt > 0) & (install_cnt > 0)).to(torch.float32)

    # Stack in a fixed order
    feat_map = {
        "num_ops_visible":            num_ops_visible,
        "multi_ops_flag":             multi_ops_flag,
        "demolish_ops_visible":       demolish_cnt,
        "install_ops_visible":        install_cnt,
        "repair_ops_visible":         repair_cnt,
        "finishing_ops_visible":      finishing_cnt,
        "kitchen_ops_visible":        kitchen_cnt,
        "structural_ops_visible":     structural_cnt,
        "demolish_completion_ratio":  demolish_ratio,
        "install_completion_ratio":   install_ratio,
        "repair_completion_ratio":    repair_ratio,
        "finishing_completion_ratio": finishing_ratio,
        "kitchen_completion_ratio":   kitchen_ratio,
        "structural_completion_ratio":structural_ratio,
        "has_complementary_demolish": has_compl_demolish,
        "has_complementary_install":  has_compl_install,
        "missing_install_after_demolish": missing_install_after_demolish,
        "has_pair_44_46":             has_pair_44_46,
        "has_pair_49_70":             has_pair_49_70,
        "has_pair_52_53":             has_pair_52_53,
        "has_pair_62_61":             has_pair_62_61,
        "has_pair_314_315":           has_pair_314_315,
        "extensive_damage":           extensive_damage,
        "minimal_damage":             minimal_damage,
        "kitchen_complexity":         kitchen_complexity,
        "has_structural_work":        has_structural_work,
        "has_rebuild_phase":          has_rebuild_phase,
    }

    sel = include if include is not None else DEFAULT_DEP_INCLUDE
    selected_tensors = [feat_map[name] for name in DEP_FEATURE_ORDER if sel.get(name, True)]

    feats = torch.stack(selected_tensors, dim=1).to(torch.float32)
    return feats

def collate_fn_infer(batch):
    B = len(batch)

    feat_dim = batch[0]["X"].shape[0] + batch[0]["room_cluster_one_hot"].shape[0]
    max_set = max(len(it["calculus"]) if it["calculus"] else 1 for it in batch)

    X = torch.empty((B, feat_dim), dtype=batch[0]["X"].dtype)
    context = torch.zeros((B, max_set, feat_dim), dtype=batch[0]["X"].dtype)
    context_mask = torch.zeros((B, max_set), dtype=torch.bool)

    project_id = torch.tensor([it["project_id"] for it in batch], dtype=torch.long)
    room = [it.get("room", "") for it in batch]
    ids = torch.tensor([it.get("id", -1) for it in batch], dtype=torch.long)

    for i, it in enumerate(batch):
        X[i] = torch.cat([it["X"], it["room_cluster_one_hot"]])
        ctx_rows = []
        for entry in it["calculus"]:
            wo = entry["work_operations_index_encoded"]
            rc = entry["room_cluster_one_hot"]
            ctx_rows.append(torch.cat([wo.detach().clone(), rc.detach().clone()]))
        if ctx_rows:
            ctx = torch.stack(ctx_rows, dim=0)
            n = ctx.shape[0]
            context[i, :n, :] = ctx
            context_mask[i, :n] = True

    recover_office_zip_code = torch.tensor([it["recover_office_zip_code"] for it in batch], dtype=torch.int32)
    damage_address_zip_code = torch.tensor([it["damage_address_zip_code"] for it in batch], dtype=torch.int32)
    office_distance        = torch.tensor([it["office_distance"] for it in batch], dtype=torch.float32)
    case_creation_year     = torch.tensor([it["case_creation_year"] for it in batch], dtype=torch.int32)
    case_creation_month    = torch.tensor([it["case_creation_month"] for it in batch], dtype=torch.int32)

    out = {
        "X": X,
        "context": context,
        "context_mask": context_mask,
        "project_id": project_id,
        "room": room,
        "ids": ids,  # -1 if not present

        "recover_office_zip_code": recover_office_zip_code,
        "damage_address_zip_code": damage_address_zip_code,
        "office_distance": office_distance,
        "case_creation_year": case_creation_year,
        "case_creation_month": case_creation_month,
    }

    out["meta"] = build_meta_features(out).to(torch.float32)  # [B, 10]
    out["deps"] = build_dependency_features(out).to(torch.float32)  # [B, 27]

    return out

def collate_fn(batch):
    B = len(batch)

    feat_dim = batch[0]["X"].shape[0] + batch[0]["room_cluster_one_hot"].shape[0]
    max_set = max(len(it["calculus"]) if it["calculus"] else 1 for it in batch)

    X = torch.empty((B, feat_dim), dtype=batch[0]["X"].dtype)
    context = torch.zeros((B, max_set, feat_dim), dtype=batch[0]["X"].dtype)
    context_mask = torch.zeros((B, max_set), dtype=torch.bool)
    Y = torch.empty((B, batch[0]["Y"].shape[0]), dtype=batch[0]["Y"].dtype)

    project_id = torch.tensor([it["project_id"] for it in batch], dtype=torch.long)
    room = [it.get("room", "") for it in batch]

    for i, it in enumerate(batch):
        X[i] = torch.cat([it["X"], it["room_cluster_one_hot"]])
        ctx_rows = []
        for entry in it["calculus"]:
            wo = entry["work_operations_index_encoded"]
            rc = entry["room_cluster_one_hot"]
            ctx_rows.append(torch.cat([wo.detach().clone(), rc.detach().clone()]))
        if ctx_rows:
            ctx = torch.stack(ctx_rows, dim=0)
            n = ctx.shape[0]
            context[i, :n, :] = ctx
            context_mask[i, :n] = True
        Y[i] = it["Y"]

    recover_office_zip_code = torch.tensor([it["recover_office_zip_code"] for it in batch], dtype=torch.int32)
    damage_address_zip_code = torch.tensor([it["damage_address_zip_code"] for it in batch], dtype=torch.int32)
    office_distance        = torch.tensor([it["office_distance"] for it in batch], dtype=torch.float32)
    case_creation_year     = torch.tensor([it["case_creation_year"] for it in batch], dtype=torch.int32)
    case_creation_month    = torch.tensor([it["case_creation_month"] for it in batch], dtype=torch.int32)

    out = {
        "X": X,
        "Y": Y,
        "context": context,
        "context_mask": context_mask,
        "project_id": project_id,
        "room": room,

        "recover_office_zip_code": recover_office_zip_code,
        "damage_address_zip_code": damage_address_zip_code,
        "office_distance": office_distance,
        "case_creation_year": case_creation_year,
        "case_creation_month": case_creation_month,
    }

    # Prebuild meta ONCE per batch (vectorized)
    out["meta"] = build_meta_features(out)  # [B, 10]
    out["deps"] = build_dependency_features(out)
    
    return out
