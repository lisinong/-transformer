#!/usr/bin/env bash
# run_ablation.sh
set -u
set -o pipefail

BASE_CFG="${1:-configs/config.yaml}"
TS=$(date +%Y%m%d_%H%M%S)

# 读取原配置中的 training.output_dir 作为“根目录”
BASE_OUTPUT_DIR=$(python - <<'PY' "$BASE_CFG"
import sys, yaml, os
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
out = (cfg.get("training") or {}).get("output_dir")
print(out if out else "")
PY
)

if [[ -z "${BASE_OUTPUT_DIR}" ]]; then
  echo "[ERR] 'training.output_dir' not found in ${BASE_CFG}. Please set it in your config."
  exit 1
fi

# 统一把所有本次消融的结果放在 原配置目录/ablation_时间戳/...
EXP_ROOT="${BASE_OUTPUT_DIR%/}/ablation_${TS}"
CFG_DIR="${EXP_ROOT}/configs"
LOG_DIR="${EXP_ROOT}/logs"
CSV="${EXP_ROOT}/summary.csv"
mkdir -p "${CFG_DIR}" "${LOG_DIR}"

# 6 runs
declare -a RUNS=(
  "pe=sinusoidal dr=0.2 nh=4 tie=false"
  "pe=learned    dr=0.2 nh=4 tie=false"
  "pe=none       dr=0.2 nh=4 tie=false"
  "pe=sinusoidal dr=0.0 nh=4 tie=false"
  "pe=sinusoidal dr=0.2 nh=2 tie=false"
  "pe=sinusoidal dr=0.2 nh=4 tie=true"
)

echo "run_id,pe,dr,nh,tie,tf_loss,tf_acc,tf_bleu,tf_rouge,tf_rep1,tf_rep2,gen_bleu,gen_rouge,gen_rep1,gen_rep2" > "${CSV}"

# 将派生配置写到 outcfg，并把其中的 training.output_dir = <BASE_OUTPUT_DIR>/ablation_TS/<run_id>
write_cfg () {
  local base="$1" out="$2" outdir="$3" pe="$4" dr="$5" nh="$6" tie="$7"
  python - "$base" "$out" "$outdir" "$pe" "$dr" "$nh" "$tie" <<'PY'
import sys, yaml, os
base, out, outdir, pe, dr, nh, tie = sys.argv[1:]
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

mcfg = cfg.setdefault("model", {})
mcfg["positional_encoding"] = pe
mcfg["dropout"] = float(dr)
mcfg["n_heads"] = int(nh)
mcfg["tie_weights"] = (tie.lower() == "true")

tcfg = cfg.setdefault("training", {})
tcfg["output_dir"] = outdir  # 关键：保存到 原配置目录/ablation_TS/<run_id>

os.makedirs(outdir, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"[CFG] wrote {out} -> training.output_dir={outdir}")
PY
}

append_metrics () {
  local run_id="$1" pe="$2" dr="$3" nh="$4" tie="$5" logf="$6"
  local tf_loss="" tf_acc="" tf_bleu="" tf_rouge="" tf_rep1="" tf_rep2=""
  local gen_bleu="" gen_rouge="" gen_rep1="" gen_rep2=""

  local tf_line
  tf_line=$(grep -E "^\[(TF|Test)\]\[Summary\]" -m1 "$logf" || true)
  if [[ -z "$tf_line" ]]; then
    tf_line=$(grep -E "^\[Test\] Loss=" -m1 "$logf" || true)
  fi
  if [[ -n "$tf_line" ]]; then
    tf_loss=$(echo "$tf_line"  | sed -n 's/.*Loss=\([0-9.]*\).*/\1/p' | head -1)
    tf_acc=$(echo "$tf_line"   | sed -n 's/.*Acc=\([0-9.]*\)%.*/\1/p' | head -1)
    tf_bleu=$(echo "$tf_line"  | sed -n 's/.*BLEU=\([0-9.]*\).*/\1/p' | head -1)
    tf_rouge=$(echo "$tf_line" | sed -n 's/.*ROUGE-L(F1)=\([0-9.]*\).*/\1/p' | head -1)
    tf_rep1=$(echo "$tf_line"  | sed -n 's/.*Repetition@1=\([0-9.]*\)%.*/\1/p' | head -1)
    tf_rep2=$(echo "$tf_line"  | sed -n 's/.*Repetition@2=\([0-9.]*\)%.*/\1/p' | head -1)
  fi

  local gen_line
  gen_line=$(grep -E "^\[GEN\]\[Summary\]" -m1 "$logf" || true)
  if [[ -n "$gen_line" ]]; then
    gen_bleu=$(echo "$gen_line"  | sed -n 's/.*BLEU=\([0-9.]*\).*/\1/p' | head -1)
    gen_rouge=$(echo "$gen_line" | sed -n 's/.*ROUGE-L(F1)=\([0-9.]*\).*/\1/p' | head -1)
    gen_rep1=$(echo "$gen_line"  | sed -n 's/.*Repetition@1=\([0-9.]*\)%.*/\1/p' | head -1)
    gen_rep2=$(echo "$gen_line"  | sed -n 's/.*Repetition@2=\([0-9.]*\)%.*/\1/p' | head -1)
  fi

  echo "${run_id},${pe},${dr},${nh},${tie},${tf_loss},${tf_acc},${tf_bleu},${tf_rouge},${tf_rep1},${tf_rep2},${gen_bleu},${gen_rouge},${gen_rep1},${gen_rep2}" >> "${CSV}"
}

echo "[INFO] Base config: ${BASE_CFG}"
echo "[INFO] Base output_dir: ${BASE_OUTPUT_DIR}"
echo "[INFO] This run root:   ${EXP_ROOT}"

i=0
for RUN in "${RUNS[@]}"; do
  i=$((i+1))
  pe="sinusoidal"; dr="0.2"; nh="4"; tie="false"
  for kv in ${RUN}; do
    k="${kv%%=*}"; v="${kv#*=}"
    case "$k" in
      pe)  pe="$v" ;;
      dr)  dr="$v" ;;
      nh)  nh="$v" ;;
      tie) tie="$v" ;;
      *)   echo "[WARN] unknown key '$k'";;
    esac
  done

  run_id="pe-${pe}_dr-${dr}_nh-${nh}_tie-${tie}"
  run_id="${run_id//./_}"

  outdir="${EXP_ROOT}/${run_id}"          # <== 位于你的配置目录之下
  outcfg="${CFG_DIR}/${run_id}.yaml"
  logfile="${LOG_DIR}/${run_id}.log"

  echo ""
  echo "========== [${i}/${#RUNS[@]}] ${run_id} =========="
  write_cfg "${BASE_CFG}" "${outcfg}" "${outdir}" "${pe}" "${dr}" "${nh}" "${tie}"

  echo "[RUN] Training..."
  if ! python train.py --config "${outcfg}" --mode train 2>&1 | tee "${logfile}"; then
    echo "[ERR] Training failed for ${run_id}, see ${logfile}"
    append_metrics "${run_id}" "${pe}" "${dr}" "${nh}" "${tie}" "${logfile}"
    continue
  fi

  echo "[RUN] Testing..."
  if ! python train.py --config "${outcfg}" --mode test 2>&1 | tee -a "${logfile}"; then
    echo "[ERR] Testing failed for ${run_id}, see ${logfile}"
    append_metrics "${run_id}" "${pe}" "${dr}" "${nh}" "${tie}" "${logfile}"
    continue
  fi

  append_metrics "${run_id}" "${pe}" "${dr}" "${nh}" "${tie}" "${logfile}"
done

echo ""
echo "[DONE] Ablation finished. Summary CSV: ${CSV}"
echo "[TIP ] View quick table:"
echo "       column -s, -t < ${CSV} | less -#2 -N"
