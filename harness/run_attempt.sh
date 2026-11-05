#!/usr/bin/env bash
# Run one blind agent attempt: fresh scratch folder, practice data, harness copy, forward-pass budget, then
# `claude -p` as a separate process with the task prompt. Usage: run_attempt.sh <attempt-name> <budget> <minutes>
set -euo pipefail
NAME=$1; BUDGET=$2; MINUTES=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=${WSP_HOME:-/home/box/wsp_agent}
PRACTICE_SRC=${PRACTICE_SRC:-$REPO/../practice}
A=$W/attempts/$NAME; S=$A/work; ST=$A/state
[ -e "$A" ] && { echo "attempt $NAME exists"; exit 1; }
mkdir -p $S/practice $ST $W/bin $W/harness
cp $REPO/harness/wsp.py $REPO/harness/build_model.py $REPO/scripts/mathgrade.py $REPO/scripts/llserver.py $W/harness/
cp $PRACTICE_SRC/gsm8k_train.jsonl $PRACTICE_SRC/math_train.jsonl $S/practice/
printf '#!/usr/bin/env bash\nexec %s/venv/bin/python %s/harness/wsp.py "$@"\n' $W $W > $W/bin/wsp; chmod +x $W/bin/wsp
for p in python python3 pip; do rm -f $W/bin/$p; printf '#!/usr/bin/env bash\nexec %s/venv/bin/%s "$@"\n' $W $p > $W/bin/$p; chmod +x $W/bin/$p; done
echo "{\"budget\": $BUDGET, \"used\": 0, \"passes\": []}" > $ST/budget.json
sed -e "s#{SCRATCH}#$S#g" -e "s#{HARNESS}#$W/harness#g" -e "s#{BASE}#$W/base#g" -e "s#{BUDGET}#$BUDGET#g" \
    -e "s#{MINUTES}#$MINUTES#g" $REPO/harness/TASK_PROMPT.md > $A/task_prompt.txt
export WSP_HOME=$W WSP_SCRATCH=$S WSP_STATE=$ST WSP_PRACTICE=$S/practice PYTHONPATH=$W/harness PATH=$W/bin:$PATH
date -u +%FT%TZ > $A/started
cd $S
# Web tools and MCP connectors are switched off; Bash network access cannot be blocked here (see README caveat).
taskset -c ${AGENT_CPUS:-5-7} timeout --signal=INT $((MINUTES * 60 + 900)) claude -p "$(cat $A/task_prompt.txt)" --dangerously-skip-permissions \
  --output-format stream-json --verbose --disallowedTools WebSearch WebFetch \
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' > $A/transcript.jsonl 2> $A/claude_stderr.log || echo "claude exit $?" >> $A/claude_stderr.log
date -u +%FT%TZ > $A/finished
echo "attempt $NAME done"
