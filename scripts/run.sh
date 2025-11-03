#!/usr/bin/env bash
# run_min_ablation.sh
# Minimal ablation on your TransformerSeq2Seq: 6 runs.

set -euo pipefail

BASE_CFG="configs/base_seq2seq.yaml"
RESULTS_TSV="ablation_results.tsv"
TMP_DIR=".ablation_tmp"
RUNS_DIR="runs"
TRAIN_SCRIPT="train_seq2seq.py"
EVAL_SCRIPT="eval_seq2seq.py"

EVAL_SPLIT="test"
DECODE="beam"
BEAM_SIZE=5
MAX_NEW_TOKENS=128

# Faster dev
MAX_STEPS_OVERRIDE=10000

# ---------- YAML override helper (reuse your original) ----------
override_yaml() {
  local base_yaml="$1"
  local out_yaml="$2"
  shift 2
  python - "$base_yaml" "$out_yaml" "$@" <<'PYCODE'
import sys, yaml
base, out = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]
cfg = yaml.safe_load(open(base, 'r')) or {}

def parse_value(v):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v

def set_by_path(d, path, value):
    cur = d
    keys = path.split('.')
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

for p in pairs:
    k, v = p.split('=', 1)
    set_by_path(cfg, k, parse_value(v))

yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False, allow_unicode=True)
PYCODE
}

align_cfg_to_ckpt() {
  local cfg_path="$1"
  local ckpt_path="$2"
  python - "$cfg_path" "$ckpt_path" <<'PY'
import sys, yaml, torch
cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path, 'r')) or {}
model_cfg = dict(cfg.get("model") or {})
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt.get("model", ckpt)
enc_key = "enc_layers.0.ffn.fc1.weight"
dec_key = "dec_layers.0.ffn.fc1.weight"
ffn_w = state.get(enc_key)
if ffn_w is None:
    ffn_w = state.get(dec_key)
if ffn_w is None:
    print("[align] no ffn.fc1.weight in ckpt; skip.")
    raise SystemExit(0)

d_ff, d_model = int(ffn_w.shape[0]), int(ffn_w.shape[1])
candidates = ["dim_feedforward", "ffn_hidden_dim", "d_ff", "mlp_dim"]
ffn_key = next((k for k in candidates if k in model_cfg), "dim_feedforward")
model_cfg[ffn_key] = d_ff
if "d_model" in model_cfg:
    model_cfg["d_model"] = d_model
cfg["model"] = model_cfg
yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False, allow_unicode=True)
print(f"[align] set {ffn_key}={d_ff}, d_model={d_model} in {cfg_path}")
PY
}

extract_rouge() {
  local line="$1"
  local r1 r2 rl n
  n=$(echo "$line" | sed -n 's/.*samples=\([0-9]\+\).*/\1/p')
  r1=$(echo "$line" | sed -n 's/.*ROUGE-1=\([0-9.]\+\).*/\1/p')
  r2=$(echo "$line" | sed -n 's/.*ROUGE-2=\([0-9.]\+\).*/\1/p')
  rl=$(echo "$line" | sed -n 's/.*ROUGE-L=\([0-9.]\+\).*/\1/p')
  echo -e "${n}\t${r1}\t${r2}\t${rl}"
}

mkdir -p "$TMP_DIR"
if [ ! -f "$RESULTS_TSV" ]; then
  echo -e "timestamp\trun_name\toptimizer\tlr\tbetas_or_momentum\tposenc\tdropout\tscheduler\taccum_steps\tmax_steps\tn_layers\ttie_weights\tsamples\tROUGE-1\tROUGE-2\tROUGE-L\tckpt" > "$RESULTS_TSV"
fi

# Common knobs
OPT="adamw"
LR="0.00001"
B1="0.9"
B2="0.98"
SCH="cosine"
ACC="1"

# Define the 6 runs
declare -a RUNS=(
  # 1) Baseline
  "pe=sinusoidal dr=0.1 nl=4 tie=false"
  # 2) Learned PE
  "pe=learned dr=0.1 nl=4 tie=false"
  # 3) No PE
  "pe=none dr=0.1 nl=4 tie=false"
  # 4) Dropout 0.0
  "pe=sinusoidal dr=0.0 nl=4 tie=false"
  # 5) Fewer layers (2)
  "pe=sinusoidal dr=0.1 nl=2 tie=false"
  # 6) Weight tying
  "pe=sinusoidal dr=0.1 nl=4 tie=true"
)

for spec in "${RUNS[@]}"; do
  # parse spec
  eval "$spec"   # sets pe, dr, nl, tie

  RUN_NAME="minabl_pe-${pe}_dr-${dr}_nl-${nl}_tie-${tie}"
  CFG_PATH="${TMP_DIR}/${RUN_NAME}.yaml"


  OV=(
    "run_name=${RUN_NAME}"
    "optim.name=${OPT}"
    "optim.lr=${LR}"
    "optim.betas=[${B1},${B2}]"
    "sched.scheduler=${SCH}"
    "train.accum_steps=${ACC}"
    "model.positional_encoding=${pe}"
    "model.dropout=${dr}"
    "model.n_layers=${nl}"
    "model.tie_weights=${tie}"
  )
  if [ -n "${MAX_STEPS_OVERRIDE:-}" ]; then OV+=("train.max_steps=${MAX_STEPS_OVERRIDE}"); fi

  override_yaml "$BASE_CFG" "$CFG_PATH" "${OV[@]}"

  echo "==== Train: ${RUN_NAME} ===="
  python "$TRAIN_SCRIPT" --config "$CFG_PATH"

  # ckpt path (按你原先的惯例)
  CKPT="${RUNS_DIR}/${RUN_NAME}/best.pt"
  [ -f "$CKPT" ] || { echo "skip eval (no ckpt)"; continue; }

  # align
  align_cfg_to_ckpt "$CFG_PATH" "$CKPT"

  echo "==== Eval: ${RUN_NAME} ===="
  EVAL_OUT=$(python "$EVAL_SCRIPT" --config "$CFG_PATH" --ckpt "$CKPT" --split "$EVAL_SPLIT" --decode "$DECODE" --beam_size "$BEAM_SIZE" --max_new_tokens "$MAX_NEW_TOKENS" || true)
  LINE=$(echo "$EVAL_OUT" | tail -n 1); read SAMPLES R1 R2 RL < <(extract_rouge "$LINE")
  TS=$(date +"%Y-%m-%d %H:%M:%S")
  MAX_STEPS=$(python - <<'PYREAD' "$CFG_PATH"
import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["train"]["max_steps"])
PYREAD
  )
  echo -e "${TS}\t${RUN_NAME}\t${OPT}\t${LR}\tbeta1=${B1},beta2=${B2}\t${pe}\t${dr}\t${SCH}\t${ACC}\t${MAX_STEPS}\t${nl}\t${tie}\t${SAMPLES}\t${R1}\t${R2}\t${RL}\t${CKPT}" >> "$RESULTS_TSV"
done

echo "Done. Results in ${RESULTS_TSV}"
