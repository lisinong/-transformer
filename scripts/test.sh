#!/usr/bin/env bash
# eval_only_ablation.sh
# Evaluate the 6 predefined runs under runs/<RUN_NAME>/ without training.
set -euo pipefail

# ===== User knobs =====
BASE_CFG="${BASE_CFG:-configs/base_seq2seq.yaml}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_seq2seq.py}"
RUNS_DIR="${RUNS_DIR:-runs}"
TMP_DIR="${TMP_DIR:-.ablation_eval_tmp}"

RESULTS_TSV="${RESULTS_TSV:-ablation_results_evalonly.tsv}"
RESULTS_XLSX="${RESULTS_XLSX:-ablation_results_evalonly.xlsx}"

EVAL_SPLIT="${EVAL_SPLIT:-test}"
DECODE="${DECODE:-beam}"          # greedy / beam
BEAM_SIZE="${BEAM_SIZE:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

# ===== The same 6 runs you trained =====
declare -a RUNS=(
  "pe=sinusoidal dr=0.1 nl=4 tie=false"
  "pe=learned    dr=0.1 nl=4 tie=false"
  "pe=none       dr=0.1 nl=4 tie=false"
  "pe=sinusoidal dr=0.0 nl=4 tie=false"
  "pe=sinusoidal dr=0.1 nl=2 tie=false"
  "pe=sinusoidal dr=0.1 nl=4 tie=true"
)

mkdir -p "$TMP_DIR"

# ---- sanity: rouge-score must exist (否则你的 eval 会静默打印0分) ----
python - <<'PY' || { echo "[ERROR] Please 'pip install rouge-score pandas openpyxl'"; exit 1; }
import importlib
for m in ("rouge_score","pandas","openpyxl"):
    try: importlib.import_module(m)
    except Exception: raise SystemExit(1)
print("[ok] deps ready")
PY

# ---- helpers ----
override_yaml() {
  local base_yaml="$1"; local out_yaml="$2"; shift 2
  python - "$base_yaml" "$out_yaml" "$@" <<'PYCODE'
import sys, yaml
base, out = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]
cfg = yaml.safe_load(open(base, 'r')) or {}
def parse_value(v):
    try: return yaml.safe_load(v)
    except Exception: return v
def set_by_path(d, path, value):
    cur = d
    for k in path.split('.')[:-1]:
        if k not in cur or not isinstance(cur[k], dict): cur[k] = {}
        cur = cur[k]
    cur[path.split('.')[-1]] = value
for p in pairs:
    k, v = p.split('=', 1)
    set_by_path(cfg, k, parse_value(v))
yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False, allow_unicode=True)
PYCODE
}

# 避免 bool(Tensor) 歧义的对齐
align_cfg_to_ckpt() {
  local cfg_path="$1"; local ckpt_path="$2"
  python - "$cfg_path" "$ckpt_path" <<'PY'
import sys, yaml, torch
cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(cfg_path, 'r')) or {}
model_cfg = dict(cfg.get("model") or {})
state = torch.load(ckpt_path, map_location="cpu")
state = state.get("model", state)

enc_key = "enc_layers.0.ffn.fc1.weight"
dec_key = "dec_layers.0.ffn.fc1.weight"
ffn_w = state.get(enc_key)
if ffn_w is None: ffn_w = state.get(dec_key)
if ffn_w is None:
    print("[align] no ffn.fc1.weight in ckpt; skip.")
    sys.exit(0)

d_ff, d_model = int(ffn_w.shape[0]), int(ffn_w.shape[1])
cands = ["dim_feedforward","ffn_hidden_dim","d_ff","mlp_dim"]
ffn_key = next((k for k in cands if k in model_cfg), "dim_feedforward")
model_cfg[ffn_key] = d_ff
if "d_model" in model_cfg:
    model_cfg["d_model"] = d_model
cfg["model"] = model_cfg
yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False, allow_unicode=True)
print(f"[align] set {ffn_key}={d_ff}, d_model={d_model} in {cfg_path}")
PY
}

# 从整段 stdout 中提取最后一次 ROUGE
parse_metrics() {
  python - <<'PY'
import re, sys, json
text = sys.stdin.read()
pat = re.compile(r'\[\w+\]\s+samples\s*=\s*(\d+).*?ROUGE-1\s*=\s*([0-9.]+).*?ROUGE-2\s*=\s*([0-9.]+).*?ROUGE-L\s*=\s*([0-9.]+)', re.S)
m = None
for x in pat.finditer(text): m = x
print(json.dumps({"samples": (int(m.group(1)) if m else ""),
                  "r1": (float(m.group(2)) if m else ""),
                  "r2": (float(m.group(3)) if m else ""),
                  "rl": (float(m.group(4)) if m else "")}))
PY
}

# 初始化结果表头
if [ ! -f "$RESULTS_TSV" ]; then
  echo -e "timestamp\trun_name\tsplit\tdecode\tbeam_size\tmax_new_tokens\tsamples\tROUGE-1\tROUGE-2\tROUGE-L\tckpt" > "$RESULTS_TSV"
fi

# ===== loop over your 6 RUN_NAMEs =====
for spec in "${RUNS[@]}"; do
  eval "$spec"   # sets pe, dr, nl, tie
  RUN_NAME="minabl_pe-${pe}_dr-${dr}_nl-${nl}_tie-${tie}"
  RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
  CKPT="${RUN_PATH}/model.pt"
  [ -f "$CKPT" ] || CKPT="${RUN_PATH}/best.pt"

  if [ ! -f "$CKPT" ]; then
    echo "[skip] ${RUN_NAME}: no model.pt/best.pt under ${RUN_PATH}"
    continue
  fi

  CFG_PATH="${TMP_DIR}/${RUN_NAME}.yaml"
  override_yaml "$BASE_CFG" "$CFG_PATH" "run_name=${RUN_NAME}"

  align_cfg_to_ckpt "$CFG_PATH" "$CKPT"

  echo "==== Eval: ${RUN_NAME} ===="
  OUT=$(python "$EVAL_SCRIPT" --config "$CFG_PATH" --ckpt "$CKPT" \
         --split "$EVAL_SPLIT" --decode "$DECODE" \
         --beam_size "$BEAM_SIZE" --max_new_tokens "$MAX_NEW_TOKENS" || true)

  METRICS=$(echo "$OUT" | parse_metrics)
  SAMPLES=$(python - <<PY "$METRICS"
import json,sys; print(json.loads(sys.argv[1])["samples"])
PY
  )
  R1=$(python - <<PY "$METRICS"
import json,sys; print(json.loads(sys.argv[1])["r1"])
PY
  )
  R2=$(python - <<PY "$METRICS"
import json,sys; print(json.loads(sys.argv[1])["r2"])
PY
  )
  RL=$(python - <<PY "$METRICS"
import json,sys; print(json.loads(sys.argv[1])["rl"])
PY
  )

  TS=$(date +"%Y-%m-%d %H:%M:%S")
  echo -e "${TS}\t${RUN_NAME}\t${EVAL_SPLIT}\t${DECODE}\t${BEAM_SIZE}\t${MAX_NEW_TOKENS}\t${SAMPLES}\t${R1}\t${R2}\t${RL}\t${CKPT}" >> "$RESULTS_TSV"
done

# ===== TSV -> Excel =====
python - <<PY "$RESULTS_TSV" "$RESULTS_XLSX" || true
import sys, pandas as pd
tsv, xlsx = sys.argv[1], sys.argv[2]
df = pd.read_csv(tsv, sep="\t")
# 可读性：按 run_name 排序
try:
    df = df.sort_values("run_name")
except Exception:
    pass
# 写 Excel
df.to_excel(xlsx, index=False)
print(f"[ok] wrote {xlsx}")
PY

echo "Done. Results in ${RESULTS_TSV} and ${RESULTS_XLSX}"
