# %%
DATA_ROOT = "data"

from dataset.hackathon import HackathonDataset
from dataset.work_operations import WorkOperationsDataset
from dataset.metadata import MetadataDataset

ds_train = HackathonDataset(split="train", download=False, root=DATA_ROOT, seed=42)
ds_val   = HackathonDataset(split="val",   download=False, root=DATA_ROOT, seed=42)

pl_train = ds_train.get_polars_dataframe()
pl_val   = ds_val.get_polars_dataframe()

print(pl_train.columns)
print(pl_train.head())


# %%
# Add this cell right after your data loading (after the first cell)
# === List all operations with cluster codes ===
import polars as pl

# First, let's see what columns we actually have
print("Available columns:")
print(pl_train.columns)
print("\nSample of work_operation column:")
print(pl_train.select("work_operation").head(10))

# Get all unique operations from the training data (just the codes for now)
all_operations_codes = (
    pl_train
    .filter(pl.col("is_hidden") == False)  # Only visible operations
    .select("work_operation")
    .unique()
    .sort("work_operation")
)

print(f"\nTotal unique operation codes: {all_operations_codes.height}")
print("\nAll Operation Codes:")
print("=" * 30)
for row in all_operations_codes.iter_rows(named=True):
    print(f"Code: {row['work_operation']:3d}")

# Get operation frequencies for analysis
op_stats = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by("work_operation")
    .len()
    .sort("len", descending=True)
    .rename({"len": "frequency"})
)

print("\nTop 20 Most Frequent Operation Codes:")
print("=" * 40)
for row in op_stats.head(20).iter_rows(named=True):
    print(f"Code: {row['work_operation']:3d} | Freq: {row['frequency']:4d}")

# Try to get operation names from the work_operations_dataset directly
print("\nTrying to get operation names from work_operations_dataset...")
try:
    # Access the work operations dataset to get the mapping
    tickets_with_names = ds_train.work_operations_dataset._load_tickets()
    print("Tickets columns:", tickets_with_names.columns)
    
    # If we have the names, create a proper mapping
    if "work_operation_cluster_name" in tickets_with_names.columns:
        op_name_mapping = (
            tickets_with_names
            .select(["work_operation_cluster_code", "work_operation_cluster_name"])
            .unique()
            .sort("work_operation_cluster_code")
        )
        
        print(f"\nAll Operations with Names ({op_name_mapping.height} total):")
        print("=" * 80)
        for row in op_name_mapping.iter_rows(named=True):
            print(f"Code: {row['work_operation_cluster_code']:3d} | {row['work_operation_cluster_name']}")
        
        # Save with names
        op_name_mapping.write_csv("all_operations_codes.csv")
        print(f"\n✅ Saved all_operations_codes.csv with {op_name_mapping.height} operations")
        
        # Join frequencies with names for better analysis
        op_stats_with_names = (
            op_stats
            .join(
                op_name_mapping.rename({"work_operation_cluster_code": "work_operation"}),
                on="work_operation",
                how="left"
            )
            .sort("frequency", descending=True)
        )
        
        print("\nTop 20 Most Frequent Operations with Names:")
        print("=" * 80)
        for row in op_stats_with_names.head(20).iter_rows(named=True):
            name = row.get('work_operation_cluster_name', 'Unknown')
            print(f"Code: {row['work_operation']:3d} | Freq: {row['frequency']:4d} | {name}")
        
        op_stats_with_names.write_csv("operation_frequencies.csv")
        print(f"\n✅ Saved operation_frequencies.csv with names")
        
    else:
        # Fallback: save just the codes
        all_operations_codes.write_csv("all_operations_codes.csv")
        op_stats.write_csv("operation_frequencies.csv")
        print(f"\n✅ Saved files with codes only (names not available)")
        
except Exception as e:
    print(f"Error accessing work_operations_dataset: {e}")
    # Fallback: save just the codes
    all_operations_codes.write_csv("all_operations_codes.csv")
    op_stats.write_csv("operation_frequencies.csv")
    print(f"\n✅ Saved files with codes only")

# %%
import polars as pl

# Target label space size (number of clusters)
n_clusters = ds_train.work_operations_dataset.num_clusters

# Room cluster frequency
room_counts = (
    pl_train
    .select(["project_id", "room_cluster"])
    .unique()
    .group_by("room_cluster")
    .len()
    .sort("len", descending=True)
)

# Operation frequency overall (visible only)
op_freq_visible = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by("work_operation")
    .len()
    .with_columns(pl.col("len").alias("visible_freq"))
    .select(["work_operation","visible_freq"])
    .sort("visible_freq", descending=True)
)

# Operation frequency by room cluster
op_room_freq = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by(["room_cluster","work_operation"])
    .len()
    .rename({"len":"visible_freq"})
)


# %%
# Build co-occurrence counts within rooms using only visible rows
visible_pairs = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by(["project_id","room"])
    .agg(pl.col("work_operation").alias("ops"))
    .select("ops")
    .explode("ops")
)

# Self-join per (project,room) to get unordered pairs
pairs = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by(["project_id","room"])
    .agg(pl.col("work_operation").alias("ops"))
    .select(["ops"])
    .with_columns(pl.arange(0, pl.len()).alias("gid"))
    .explode("ops")
    .rename({"ops":"a"})
    .join(
        _ := (
            pl_train
            .filter(pl.col("is_hidden") == False)
            .group_by(["project_id","room"])
            .agg(pl.col("work_operation").alias("ops"))
            .select(["ops"])
            .with_columns(pl.arange(0, pl.len()).alias("gid"))
            .explode("ops")
            .rename({"ops":"b"})
        ),
        on="gid",
        how="inner",
    )
    .filter(pl.col("a") < pl.col("b"))  # unordered pairs
    .group_by(["a","b"])
    .len()
    .rename({"len":"co_count"})
)

# Marginals
marg = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by("work_operation")
    .len()
    .rename({"work_operation":"op","len":"count"})
)

# Optional: PMI = log( P(a,b) / (P(a)P(b)) )
total_rooms = (
    pl_train
    .select(["project_id","room"])
    .unique()
    .height
)

pairs_pmi = (
    pairs
    .join(marg.rename({"op":"a","count":"ca"}), on="a")
    .join(marg.rename({"op":"b","count":"cb"}), on="b")
    .with_columns([
        (pl.col("co_count") / total_rooms).alias("p_ab"),
        (pl.col("ca") / total_rooms).alias("p_a"),
        (pl.col("cb") / total_rooms).alias("p_b"),
    ])
    .with_columns( (pl.col("p_ab") / (pl.col("p_a")*pl.col("p_b"))).alias("lift") )
    .with_columns( pl.when(pl.col("p_ab")>0).then(pl.col("lift").log()).otherwise(pl.lit(0.0)).alias("pmi") )
    .select(["a","b","co_count","lift","pmi"])
)

# %%
# For each (project, room), build a set of visible ops from OTHER rooms in the same project (context)
visible_per_room = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by(["project_id","room"])
    .agg(pl.col("work_operation").alias("ops_visible"))
)

project_visible = (
    visible_per_room
    .group_by("project_id")
    .agg(pl.col("ops_visible").alias("all_room_ops"))
)

room_with_context = (
    visible_per_room
    .join(project_visible, on="project_id", how="left")
    .with_columns(
        pl.struct(["ops_visible","all_room_ops"]).map_elements(
            lambda d: list({o for ops in d["all_room_ops"] for o in ops} - set(d["ops_visible"]))
        ).alias("ctx_ops_visible_other_rooms")
    )
    .drop("all_room_ops")
)

# Room cluster counts present in the project (excluding the target room)
roomcluster_counts = (
    pl_train
    .filter(pl.col("is_hidden") == False)
    .group_by(["project_id","room_cluster"])
    .agg(pl.count().alias("cnt"))
    .pivot(values="cnt", index="project_id", columns="room_cluster")
    .fill_null(0)
)

# %%
# Metadata already parsed/encoded by MetadataDataset (incl. office_distance commas->dot)
meta_pl = ds_train.metadata_dataset.data  # includes insurance_company_one_hot etc. 
# Columns: insurance_company_one_hot, (zip codes as strings), office_distance(float), case_creation_year, case_creation_month
# We’ll derive:
meta_feats = (
    meta_pl
    .with_columns([
        pl.col("damage_address_zip_code").cast(pl.Utf8).str.slice(0,2).alias("damage_zip_p2"),
        pl.col("recover_office_zip_code").cast(pl.Utf8).str.slice(0,2).alias("office_zip_p2"),
        (pl.col("recover_office_zip_code") == pl.col("damage_address_zip_code")).cast(pl.Int8).alias("same_zip"),
        # cyclical month; cast month to int if needed
        pl.col("case_creation_month").cast(pl.Int32).alias("m_int"),
    ])
    .with_columns([
        ( (2*3.14159265*pl.col("m_int")/12).sin() ).alias("m_sin"),
        ( (2*3.14159265*pl.col("m_int")/12).cos() ).alias("m_cos"),
    ])
    .drop("m_int")
)


# %%
# Start with unique (project, room, room_cluster) rows to represent each target room
target_rooms = (
    pl_train
    .select(["project_id","room","room_cluster"])
    .unique()
)

tbl = (
    target_rooms
    .join(visible_per_room, on=["project_id","room"], how="left")
    .join(room_with_context, on=["project_id","room"], how="left")
    .join(meta_feats, on="project_id", how="left")
)

# Number of companies for the one-hot length (from your dataset)
n_companies = ds_train.metadata_dataset.num_companies  # available in MetadataDataset

tbl = tbl.with_columns([
    # Empty list for List[Int64] columns
    pl.when(pl.col("ops_visible").is_null())
      .then(pl.lit([], dtype=pl.List(pl.Int64)))
      .otherwise(pl.col("ops_visible"))
      .alias("ops_visible"),

    pl.when(pl.col("ctx_ops_visible_other_rooms").is_null())
      .then(pl.lit([], dtype=pl.List(pl.Int64)))
      .otherwise(pl.col("ctx_ops_visible_other_rooms"))
      .alias("ctx_ops_visible_other_rooms"),

    # Zero vector for the one-hot List[Int8]
    pl.when(pl.col("insurance_company_one_hot").is_null())
      .then(pl.lit([0]*n_companies, dtype=pl.List(pl.Int8)))
      .otherwise(pl.col("insurance_company_one_hot"))
      .alias("insurance_company_one_hot"),
])


# %%
# === 3.7: Candidate generation (memory-safe, version-safe, None-safe) ===
import polars as pl
import math, gc

# ---- Parameters
K = 20                 # start small; you can increase later
BATCH_ROOMS = 50_000   # explode rooms in batches to keep RAM low
WO_DTYPE = pl.Int64    # dtype for work_operation codes; change to Int32 if your data uses that

# ---- 3.7.1 Build global and per-room-cluster top lists (cheap)
top_global = (
    op_freq_visible
    .select("work_operation")
    .head(K)["work_operation"].to_list()
)
top_global = [int(x) for x in top_global if x is not None]  # ensure pure Python ints

room_clusters = op_room_freq.select("room_cluster").unique()["room_cluster"].to_list()
top_by_room = {
    rc: (
        op_room_freq
        .filter(pl.col("room_cluster") == rc)
        .sort("visible_freq", descending=True)
        .select("work_operation")
        .head(K)["work_operation"].to_list()
    )
    for rc in room_clusters
}
top_by_room = {rc: [int(x) for x in xs if x is not None] for rc, xs in top_by_room.items()}

# ---- Helpers (version-safe + None-safe)
def _norm_list(xs):
    """Return list of ints; skip None and non-castables."""
    if xs is None:
        return []
    out = []
    for x in xs:
        if x is None:
            continue
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            # skip weird values
            continue
    return out

def _uniq_preserve(xs):
    seen = set()
    out = []
    for x in _norm_list(xs):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _list_diff(a, b):
    sb = set(_norm_list(b))
    return [x for x in _norm_list(a) if x not in sb]

# ---- 3.7.2 Build candidate lists per (project, room) WITHOUT exploding
# Combine: context ops (other rooms in project) + top_global + top_by_room[room_cluster]
# Then remove ops already visible in the room (leakage-safe).
tbl_small = (
    tbl.lazy()
      .with_columns([
          pl.lit(top_global, dtype=pl.List(WO_DTYPE)).alias("cand_global"),
          pl.col("room_cluster").map_elements(
              lambda rc: top_by_room.get(rc, []),
              return_dtype=pl.List(WO_DTYPE),
          ).alias("cand_room"),
      ])
      # concat lists -> unique (order-preserving) via typed UDF, None-safe
      .with_columns(
          pl.struct(["ctx_ops_visible_other_rooms", "cand_global", "cand_room"])
            .map_elements(
                lambda d: _uniq_preserve(
                    _norm_list(d["ctx_ops_visible_other_rooms"])
                    + _norm_list(d["cand_global"])
                    + _norm_list(d["cand_room"])
                ),
                return_dtype=pl.List(WO_DTYPE),
            )
            .alias("candidates")
      )
      # typed UDF list difference (since .arr.set_difference may not exist); None-safe
      .with_columns(
          pl.struct(["candidates", "ops_visible"]).map_elements(
              lambda d: _list_diff(d["candidates"], d["ops_visible"]),
              return_dtype=pl.List(WO_DTYPE),
          ).alias("candidates")
      )
)

# ---- 3.7.3 Materialize and batch-explode to avoid RAM spikes
tbl_small = tbl_small.collect().with_row_count("rid")

rooms = tbl_small.select(["project_id", "room", "rid"])
n = rooms.height
batch_size = BATCH_ROOMS
n_batches = math.ceil(n / batch_size)

exploded_parts = []
for b in range(n_batches):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, n)
    ids = rooms.slice(lo, hi - lo).select("rid")

    part = (
        tbl_small.join(ids, on="rid", how="inner")
                 .select(["project_id", "room", "ops_visible", "candidates"])
                 .explode("candidates")
                 .rename({"candidates": "cand_code"})
                 .with_columns(pl.col("cand_code").cast(WO_DTYPE))
    )
    exploded_parts.append(part)
    del part
    gc.collect()

candidates = pl.concat(exploded_parts, how="vertical")
del exploded_parts
gc.collect()

print("Rooms:", n, "| Candidate rows:", candidates.height)
print(candidates.head())


# %%
# === 3.8: Features + labels + quick baseline, score, and CSV feature analysis ===
from metrics.score import normalized_rooms_score
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------- SAFE-MODE TOGGLES ----------
SAFE_MAX_ROWS = 400_000   # cap train rows for sklearn; set None to use all
USE_SGD = False           # True => SGDClassifier (very light); False => LogisticRegression
K_PRED = 20               # top-K per room (you can tune this later)

# ---------- 1) Popularity prior from tickets ----------
tickets = (
    ds_train.work_operations_dataset
    ._load_tickets()
    .select(
        pl.col("work_operation_cluster_code").alias("cand_code"),
        pl.col("normalized_n_tickets").alias("cand_pop"),
    )
)

cand_feats = (
    candidates
    .join(tickets, on="cand_code", how="left")
    .with_columns(pl.col("cand_pop").fill_null(0.0))
)

# ---------- 2) Join meta + build engineered features ----------
# Pull full zips too so we can compute first-2-digit match (|a-b| < 100)
want_cols = [
    "project_id","room","room_cluster",
    "m_sin","m_cos","office_distance",
    "insurance_company_one_hot",
    "case_creation_year","case_creation_month",
    "same_zip","damage_zip_p2","office_zip_p2",
    "damage_address_zip_code","recover_office_zip_code",  # NEW
]
have_cols = [c for c in want_cols if c in tbl.columns]
light_meta = tbl.select(have_cols)

cand_feats = cand_feats.join(light_meta, on=["project_id","room"], how="left")

# Ensure year/month are ints for arithmetic
cand_feats = cand_feats.with_columns([
    pl.col("case_creation_year").cast(pl.Int32),
    pl.col("case_creation_month").cast(pl.Int32),
])

# Parse full zips to Int32 (safe)
cand_feats = cand_feats.with_columns([
    pl.col("damage_address_zip_code").cast(pl.Utf8).map_elements(
        lambda s: int("".join(ch for ch in s if ch.isdigit())[:4]) if (s and any(ch.isdigit() for ch in s)) else -1,
        return_dtype=pl.Int32,
    ).alias("damage_zip_int"),
    pl.col("recover_office_zip_code").cast(pl.Utf8).map_elements(
        lambda s: int("".join(ch for ch in s if ch.isdigit())[:4]) if (s and any(ch.isdigit() for ch in s)) else -1,
        return_dtype=pl.Int32,
    ).alias("office_zip_int"),
])

# Same-area by first 2 digits: |a - b| < 100
cand_feats = cand_feats.with_columns(
    (
        (pl.col("damage_zip_int") != -1) &
        (pl.col("office_zip_int") != -1) &
        ((pl.col("damage_zip_int") - pl.col("office_zip_int")).abs() < 100)
    ).cast(pl.Int8).alias("same_area_zip2")
)

# Month index for temporal proximity
cand_feats = cand_feats.with_columns(
    (pl.col("case_creation_year") * 12 + pl.col("case_creation_month")).alias("month_index")
)

cand_feats = cand_feats.with_columns([
    # Season (quarters)
    ((pl.col("case_creation_month") - 1) // 3).alias("season"),
    
    # Is winter? (higher claim season)
    pl.col("case_creation_month").is_in([11, 12, 1, 2]).cast(pl.Int8).alias("is_winter"),
    
    # Days since epoch (for trend analysis)
    (pl.col("case_creation_year") * 365 + pl.col("case_creation_month") * 30).alias("days_since_epoch")
])

# Add this right before your existing cand_feats feature engineering (before the "Room damage proxy" section)

# === DOMAIN-SPECIFIC DEPENDENCY FEATURES ===
print("Creating domain-specific dependency features...")

# First, let's analyze the operation pairs from your data to understand real dependencies
operation_pairs = (
    pairs_pmi  # Use the PMI analysis you already created
    .sort("pmi", descending=True)
    .head(100)  # Top 100 operation pairs by PMI
)

# Define operation families based on construction workflows
OPERATION_FAMILIES = {
    # Demolition operations (commonly start workflow)
    'demolish': [44, 49, 52, 5, 62, 58, 63, 48, 314, 56, 160],  # Based on "Riv" operations
    
    # Installation/New operations (commonly follow demolish)
    'install': [46, 70, 53, 61, 11, 74, 315, 65, 162],  # Based on "Ny" operations
    
    # Remount/Repair operations
    'repair': [156, 166, 173, 175, 177, 185, 165],  # Based on "Remont" operations
    
    # Cleaning/Finishing operations (commonly at end)
    'finishing': [104, 108, 136, 257, 256, 204],  # Cleaning, protection, moving
    
    # Kitchen-specific operations
    'kitchen': [151, 153, 161, 164, 168, 170, 182],  # Demont operations
    
    # Structural operations
    'structural': [55, 313, 386],  # Cutting, special operations
}

# Create complementary operation features
def create_dependency_features(cand_feats):
    print("  → Creating num_ops_visible first...")
    
    # Create num_ops_visible if it doesn't exist
    if "num_ops_visible" not in cand_feats.columns:
        cand_feats = cand_feats.with_columns(
            pl.col("ops_visible").list.len().alias("num_ops_visible")
        )
    
    print("  → Creating operation family features...")
    
    # For each operation family, create features
    for family_name, ops in OPERATION_FAMILIES.items():
        cand_feats = cand_feats.with_columns([
            # Is candidate from this family?
            pl.col("cand_code").is_in(ops).cast(pl.Int8).alias(f"is_{family_name}_op"),
            
            # How many ops from this family are already visible?
            pl.col("ops_visible").list.eval(pl.element().is_in(ops)).list.sum().alias(f"{family_name}_ops_visible"),
            
            # Family completion ratio
            (pl.col("ops_visible").list.eval(pl.element().is_in(ops)).list.sum() / len(ops)).alias(f"{family_name}_completion_ratio")
        ])
    
    print("  → Creating workflow dependency features...")
    
    # Demolish -> Install workflow dependencies
    demolish_ops = OPERATION_FAMILIES['demolish']
    install_ops = OPERATION_FAMILIES['install']
    
    cand_feats = cand_feats.with_columns([
        # If we see demolish operations, predict install operations (high dependency)
        pl.when(pl.col("cand_code").is_in(install_ops))
        .then(
            pl.col("ops_visible").list.eval(pl.element().is_in(demolish_ops)).list.any().cast(pl.Int8)
        )
        .otherwise(0)
        .alias("has_complementary_demolish"),
        
        # If we see install operations, predict finishing operations
        pl.when(pl.col("cand_code").is_in(OPERATION_FAMILIES['finishing']))
        .then(
            pl.col("ops_visible").list.eval(pl.element().is_in(install_ops)).list.any().cast(pl.Int8)
        )
        .otherwise(0)
        .alias("has_complementary_install"),
        
        # Missing workflow steps (if we have demolish but no install yet)
        pl.when(
            pl.col("ops_visible").list.eval(pl.element().is_in(demolish_ops)).list.any() &
            ~pl.col("ops_visible").list.eval(pl.element().is_in(install_ops)).list.any()
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("missing_install_after_demolish"),
    ])
    
    print("  → Creating specific operation pair features...")
    
    # Specific high-PMI operation pairs (you'd customize this based on your PMI analysis)
    common_pairs = [
        (44, 46),   # Riv flytende gulv -> Ny gulv
        (49, 70),   # Riv gulvlist -> Ny gulvlist  
        (52, 53),   # Riv underlagsmateriale -> Ny underlag
        (62, 61),   # Riv dampsperre -> Ny dampsperre
        (314, 315), # Riv veggisolasjon -> Ny isolasjon i vegg
    ]
    
    for demolish_op, install_op in common_pairs:
        pair_support = (
            cand_feats
            .filter(
                pl.col("ops_visible").list.contains(demolish_op) &
                pl.col("ops_visible").list.contains(install_op)
            )
            .height
        )
        
        if pair_support < 100:  # Skip rare pairs
            continue
        cand_feats = cand_feats.with_columns([
            # If we see the demolish operation, strongly predict the install operation
            pl.when(pl.col("cand_code") == install_op)
            .then(pl.col("ops_visible").list.contains(demolish_op).cast(pl.Int8))
            .otherwise(0)
            .alias(f"has_pair_{demolish_op}_{install_op}"),
        ])
    
    print("  → Creating damage scope features...")
    
    # Damage scope indicators (now num_ops_visible exists)
    cand_feats = cand_feats.with_columns([
        # Extensive damage indicators (more operations = more severe)
        (pl.col("num_ops_visible") > 5).cast(pl.Int8).alias("extensive_damage"),
        (pl.col("num_ops_visible") == 1).cast(pl.Int8).alias("minimal_damage"),
        
        # Kitchen complexity (kitchens have many specialized operations)
        pl.when(pl.col("room_cluster").str.contains("(?i)kjøkken"))
        .then(pl.col("kitchen_ops_visible"))
        .otherwise(0)
        .alias("kitchen_complexity"),
        
        # Structural work indicator (walls, major repairs)
        (pl.col("structural_ops_visible") > 0).cast(pl.Int8).alias("has_structural_work"),
        
        # Work phase completeness
        pl.when(pl.col("demolish_ops_visible") > 0)
        .then(pl.col("install_ops_visible") > 0)
        .otherwise(False)
        .cast(pl.Int8)
        .alias("has_rebuild_phase")
    ])
    
    return cand_feats

# Apply the dependency features
cand_feats = create_dependency_features(cand_feats)
print("✅ Created domain-specific dependency features")

# Update your train_cols_keep to include the new features
new_dependency_features = [
    # Operation family features
    "is_demolish_op", "demolish_ops_visible", "demolish_completion_ratio",
    "is_install_op", "install_ops_visible", "install_completion_ratio", 
    "is_repair_op", "repair_ops_visible", "repair_completion_ratio",
    "is_finishing_op", "finishing_ops_visible", "finishing_completion_ratio",
    "is_kitchen_op", "kitchen_ops_visible", "kitchen_completion_ratio",
    "is_structural_op", "structural_ops_visible", "structural_completion_ratio",
    
    # Workflow dependencies
    "has_complementary_demolish", "has_complementary_install", 
    "missing_install_after_demolish",
    
    # Specific pairs (you'd add all pairs you define)
    "has_pair_44_46", "has_pair_49_70", "has_pair_52_53", 
    "has_pair_62_61", "has_pair_314_315",
    
    # Damage scope
    "extensive_damage", "minimal_damage", "kitchen_complexity",
    "has_structural_work", "has_rebuild_phase"
]

# Room damage proxy (now simplified since num_ops_visible already exists)
cand_feats = cand_feats.with_columns([
    pl.col("num_ops_visible").cast(pl.Float32),
    (pl.col("num_ops_visible") > 2).cast(pl.Int8).alias("multi_ops_flag")
])

# ---------- 2b) Event intensity: count cases per (zip2, month), then 7-month centered rolling sum ----------
# Use the damage ZIP2 as the area index: zip2 = floor(damage_zip_int / 100)
cand_feats = cand_feats.with_columns(
    (pl.when(pl.col("damage_zip_int") >= 0).then(pl.col("damage_zip_int") // 100).otherwise(-1)).alias("damage_zip2")
)


# ---------- 3) Labels for TRAIN (true missing Y) ----------
hidden_map_train = (
    ds_train.work_operations_dataset.data
      .select(["project_id","room","Y"])
      .rename({"Y":"y_codes"})          # list[int]
)

cand_train = (
    cand_feats
    .join(hidden_map_train, on=["project_id","room"], how="left")
    .with_columns(pl.col("y_codes").fill_null([]))
    .with_columns(
        pl.col("y_codes").list.contains(pl.col("cand_code")).cast(pl.Int8).alias("label")
    )
    .drop("y_codes")
)

# ---------- 4) Light casts to cut RAM ----------
float_cols = [c for c in [
    "cand_pop","m_sin","m_cos","office_distance","case_creation_year",
    "case_creation_month","same_zip","zip2_evt_7m"
] if c in cand_train.columns]
if float_cols:
    cand_train = cand_train.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])

print("cand_train rows (pre-sample):", cand_train.height)

# ---------- 5) Optional downsampling for sklearn ----------
if SAFE_MAX_ROWS is not None and cand_train.height > SAFE_MAX_ROWS:
    cand_train_small = cand_train.sample(n=SAFE_MAX_ROWS, shuffle=True, seed=42)
else:
    cand_train_small = cand_train

# ---------- 6) Minimal flattening for sklearn (version-proof) ----------
# Ensure one-hot is stored as List[Int8]
if "insurance_company_one_hot" in cand_train_small.columns:
    cand_train_small = cand_train_small.with_columns(
        pl.col("insurance_company_one_hot").cast(pl.List(pl.Int8))
    )

n_companies = ds_train.metadata_dataset.num_companies

def flatten_for_sklearn(df: pl.DataFrame) -> pl.DataFrame:
    out = df

    # # Expand insurance_company_one_hot (list[int8]) -> ic_0..ic_{n-1}
    # if "insurance_company_one_hot" in out.columns:
    #     def _get_i(xs, i):
    #         if xs is None:
    #             return 0
    #         try:
    #             v = xs[i] if i < len(xs) else 0
    #             return int(v) if v is not None else 0
    #         except Exception:
    #             return 0
    #     for i in range(n_companies):
    #         out = out.with_columns(
    #             pl.col("insurance_company_one_hot")
    #               .map_elements(lambda xs, _i=i: _get_i(xs, _i), return_dtype=pl.Int8)
    #               .alias(f"ic_{i}")
    #         )
    #     out = out.drop("insurance_company_one_hot")

    # Encode room_cluster to numeric id
    if "room_cluster" in out.columns:
        rc_uni = out.select("room_cluster").unique()["room_cluster"].to_list()
        rc_codes = {rc: i for i, rc in enumerate(rc_uni)}
        out = out.with_columns(
            pl.col("room_cluster")
              .map_elements(lambda x: rc_codes.get(x, -1), return_dtype=pl.Int32)
              .alias("room_cluster_id")
        )

    # Zip prefixes to ints (or -1 if missing) via tolerant UDF
    for zc in ("damage_zip_p2","office_zip_p2"):
        if zc in out.columns:
            out = out.with_columns(
                pl.col(zc).cast(pl.Utf8).map_elements(
                    lambda s: int("".join(ch for ch in s if ch.isdigit())[:2]) if (s and any(ch.isdigit() for ch in s)) else -1,
                    return_dtype=pl.Int32,
                ).alias(zc)
            )

    # Coerce numerics to float32
    coerce = []
    for c in ("m_sin","m_cos","office_distance","case_creation_year",
              "case_creation_month","same_zip","room_cluster_id","zip2_evt_7m"):
        if c in out.columns:
            coerce.append(pl.col(c).cast(pl.Float32))
    if coerce:
        out = out.with_columns(coerce)

    return out

# Columns to keep for sklearn matrix + identifiers
# train_cols_keep = [
#     "cand_pop","m_sin","m_cos","office_distance","case_creation_year","case_creation_month",
#     "same_zip","damage_zip_p2","office_zip_p2","room_cluster","insurance_company_one_hot",
#     "same_area_zip2","num_ops_visible","multi_ops_flag",
#     # Enhanced temporal-spatial features:
#     "season", "is_winter",
#     # NEW: Domain-specific dependency features
#     "is_demolish_op", "demolish_ops_visible", "demolish_completion_ratio",
#     "is_install_op", "install_ops_visible", "install_completion_ratio", 
#     "is_repair_op", "repair_ops_visible", "repair_completion_ratio",
#     "is_finishing_op", "finishing_ops_visible", "finishing_completion_ratio",
#     "is_kitchen_op", "kitchen_ops_visible", "kitchen_completion_ratio",
#     "is_structural_op", "structural_ops_visible", "structural_completion_ratio",
#     "has_complementary_demolish", "has_complementary_install", 
#     "missing_install_after_demolish", "has_pair_44_46", "has_pair_49_70", 
#     "has_pair_52_53", "has_pair_62_61", "has_pair_314_315",
#     "extensive_damage", "minimal_damage", "kitchen_complexity",
#     "has_structural_work", "has_rebuild_phase",
#     # identifiers we need later:
#     "project_id","room","cand_code","ops_visible","label"
# ]

train_cols_keep = [
    "cand_pop","m_sin","m_cos","office_distance","case_creation_year","case_creation_month",
    "same_zip","damage_zip_p2","office_zip_p2","room_cluster",
    "same_area_zip2","num_ops_visible","multi_ops_flag",
    "season", "is_winter",
    # Keep only the good dependency features with AUC > 0.55
    "has_rebuild_phase",        # AUC: 0.650
    "install_ops_visible",      # AUC: 0.663  
    "demolish_ops_visible",     # AUC: 0.663
    "extensive_damage",         # AUC: 0.591
    "finishing_ops_visible",    # AUC: 0.523
    "structural_ops_visible",   # AUC: 0.516
    # Remove all the problematic features:
    # "is_demolish_op", "is_install_op", "is_structural_op", etc.
    # "has_pair_*" features
    # "kitchen_*" features  
    # identifiers we need later:
    "project_id","room","cand_code","ops_visible","label"
]

train_cols_keep_updated = train_cols_keep + [
    feat for feat in new_dependency_features 
    if feat not in train_cols_keep  # Avoid duplicates
]

train_cols_keep = [c for c in train_cols_keep_updated if c in cand_train_small.columns]
Xy_pl = flatten_for_sklearn(cand_train_small.select(train_cols_keep))

# ---------- 7) Impute NaNs/nulls deterministically (by dtype) ----------
sch = Xy_pl.schema  # {col: dtype}

id_cols   = ["project_id","room","cand_code"]
non_feat  = set(id_cols + ["ops_visible","label","room_cluster"])

feat_cols = [c for c in Xy_pl.columns if c not in non_feat]

float_feats = [c for c in feat_cols if sch[c] in (pl.Float32, pl.Float64)]
int_feats   = [c for c in feat_cols if sch[c] in (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
)]

if float_feats:
    Xy_pl = Xy_pl.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0).alias(c) for c in float_feats])
if int_feats:
    Xy_pl = Xy_pl.with_columns([pl.col(c).fill_null(0).alias(c) for c in int_feats])

# Keep only numeric features
feat_cols = [c for c in feat_cols if Xy_pl.schema[c] in (
    pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
)]

# ---------- 8) Numpy arrays ----------
y = Xy_pl.select("label").to_numpy().ravel().astype(np.int32)
X = Xy_pl.select(feat_cols).to_numpy().astype(np.float32)
X[np.isnan(X)] = 0.0
X[np.isinf(X)] = 0.0

print("Feature matrix:", X.shape, "| positives:", int(y.sum()), "| negatives:", int((y==0).sum()))

# ---------- 9) Train tiny baseline ----------
if USE_SGD:
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", max_iter=10, class_weight="balanced", random_state=42)
else:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")

clf.fit(X, y)
print("Model coefficients (first 10):", clf.coef_.ravel()[:10])
print("Feature names (first 10):", feat_cols[:10])
print("Non-zero coefficients:", np.sum(clf.coef_.ravel() != 0))

# ---------- 10) Score candidates & build per-room predictions (top-K) ----------
scores = clf.predict_proba(X)[:, 1] if not USE_SGD else clf.decision_function(X)
if USE_SGD:
    scores = 1.0 / (1.0 + np.exp(-scores))  # logistic

scored = Xy_pl.select(id_cols).with_columns(pl.Series("score", scores))

print("Score distribution:")
print(f"Min: {scores.min():.6f}, Max: {scores.max():.6f}, Mean: {scores.mean():.6f}")
print(f"Unique scores: {len(np.unique(scores))}")

top_preds_sample = (
    scored
    .sort(["project_id","room","score"], descending=[False, False, True])
    .head(20)
)
print("Top 20 predictions:")
print(top_preds_sample)

preds = (
    scored
    .sort(["project_id","room","score"], descending=[False, False, True])
    .group_by(["project_id","room"])
    .head(K_PRED)
    .select(["project_id","room","cand_code"])
)

# ---------- 11) TRAIN score (aligned) ----------
pred_keys = preds.select(["project_id","room"]).unique()
tgt_keys  = hidden_map_train.select(["project_id","room"]).unique()
keys = pl.concat([pred_keys, tgt_keys], how="vertical").unique().sort(["project_id","room"])

targets_by_room = (
    hidden_map_train.group_by(["project_id","room"])
    .agg(pl.col("y_codes").first().alias("codes"))
)
preds_by_room = (
    preds.group_by(["project_id","room"])
    .agg(pl.col("cand_code").alias("codes"))
)

targets_joined = keys.join(targets_by_room, on=["project_id","room"], how="left") \
                     .with_columns(pl.col("codes").fill_null(pl.lit([], dtype=pl.List(pl.Int64))))
preds_joined   = keys.join(preds_by_room,   on=["project_id","room"], how="left") \
                     .with_columns(pl.col("codes").fill_null(pl.lit([], dtype=pl.List(pl.Int64))))

train_Y     = targets_joined.get_column("codes").to_list()
train_preds = preds_joined.get_column("codes").to_list()

print("TRAIN score (sanity):", normalized_rooms_score(train_preds, train_Y))

# ---------- 12) Feature analysis CSV ----------
rows = []
coefs = clf.coef_.ravel() if not USE_SGD else np.zeros(len(feat_cols), dtype=np.float32)  # SGD doesn't expose coef_ the same way
for i, f in enumerate(feat_cols):
    try:
        auc = roc_auc_score(y, Xy_pl.select(f).to_numpy().ravel())
    except ValueError:
        auc = np.nan
    rows.append({"feature": f, "coef": float(coefs[i]) if i < len(coefs) else np.nan, "auc": float(auc)})
fa = pd.DataFrame(rows)
fa["abs_coef"] = fa["coef"].abs()
fa.sort_values("abs_coef", ascending=False).to_csv("feature_analysis.csv", index=False)
print("✅ Saved feature_analysis.csv (columns: feature, coef, auc, abs_coef)")


# %%
import polars as pl

# Ground truth hidden ops per (project, room)
hidden_map = (
    ds_train.work_operations_dataset.data
      .select(["project_id","room","Y"])
      .rename({"Y": "y_codes"})  # list[int]
)

# Ensure cand_code dtype matches y_codes elements (usually Int64)
cand_train = (
    cand_feats
      .with_columns(pl.col("cand_code").cast(pl.Int64))
      .join(hidden_map, on=["project_id","room"], how="left")
      .with_columns(pl.col("y_codes").fill_null([]))
      .with_columns(
          pl.col("y_codes").list.contains(pl.col("cand_code")).cast(pl.Int8).alias("label")
      )
      .drop("y_codes")
)

print(cand_train.head())


# %%
# === Feature analysis (place right after clf.fit) ===
import pandas as pd

# 1) Logistic Regression coefficients (interpretability)
coef_df = pd.DataFrame({
    "feature": feat_cols,
    "coef": clf.coef_.ravel()
}).assign(abs_coef=lambda d: d["coef"].abs()).sort_values("abs_coef", ascending=False)

print("Top 50 by |coef|:")
print(coef_df.head(50))

# 2) Univariate discrimination (AUC) per feature (fast sanity)
from sklearn.metrics import roc_auc_score
uni_auc = []
for f in feat_cols:
    try:
        uni_auc.append((f, roc_auc_score(y, Xy_pl.select(f).to_numpy().ravel())))
    except ValueError:
        pass
uni_auc = pd.DataFrame(uni_auc, columns=["feature","auc"]).sort_values("auc", ascending=False)
print("\nTop 50 univariate AUCs:")
print(uni_auc.head(50))

# If you want Mutual Information too (optional, a bit slower)
# from sklearn.feature_selection import mutual_info_classif
# mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
# mi_df = pd.DataFrame({"feature": feat_cols, "MI": mi}).sort_values("MI", ascending=False)
# print("\nTop 10 MI:")
# print(mi_df.head(10))


# %%
# === Feature analysis and export (after clf.fit) ===
import pandas as pd
from sklearn.metrics import roc_auc_score

rows = []
for f in feat_cols:
    try:
        auc = roc_auc_score(y, Xy_pl.select(f).to_numpy().ravel())
    except ValueError:
        auc = np.nan
    rows.append({
        "feature": f,
        "coef": clf.coef_.ravel()[feat_cols.index(f)],
        "auc": auc,
    })

df_metrics = pd.DataFrame(rows)
df_metrics["abs_coef"] = df_metrics["coef"].abs()
df_metrics = df_metrics.sort_values("abs_coef", ascending=False)

# Save full table
df_metrics.to_csv("feature_analysis.csv", index=False)
print("✅ Saved full feature analysis to feature_analysis.csv")

# Print top section for quick view
print(df_metrics.head(50))


