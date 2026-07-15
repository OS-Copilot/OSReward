#!/bin/bash
# Start trajectory collection instead of evaluation.
# Run inside the arena container (or from /client after VM is up).

agent="navi"
model="gpt-4-vision-preview"
som_origin="oss"
a11y_backend="uia"
questions_path="collection_examples/questions.json"
output_dir="./collection_results"
max_steps=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            agent=$2
            shift 2
            ;;
        --model)
            model=$2
            shift 2
            ;;
        --som-origin)
            som_origin=$2
            shift 2
            ;;
        --a11y-backend)
            a11y_backend=$2
            shift 2
            ;;
        --questions-path)
            questions_path=$2
            shift 2
            ;;
        --output-dir)
            output_dir=$2
            shift 2
            ;;
        --max-steps)
            max_steps=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --agent <agent>"
            echo "  --model <model>"
            echo "  --som-origin <oss|a11y|...>"
            echo "  --a11y-backend <uia|win32>"
            echo "  --questions-path <json>"
            echo "  --output-dir <dir>"
            echo "  --max-steps <n>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd /client 2>/dev/null || cd "$(dirname "$0")/client"

echo "Collecting with agent=$agent model=$model questions=$questions_path"
python run_collect.py \
  --agent_name "$agent" \
  --model "$model" \
  --som_origin "$som_origin" \
  --a11y_backend "$a11y_backend" \
  --questions_path "$questions_path" \
  --output_dir "$output_dir" \
  --max_steps "$max_steps"
