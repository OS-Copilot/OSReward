# Windows Agent Arena — Trajectory Collection

[English](README.md) | [中文](README_zh.md)

Trajectory collection toolkit adapted from Windows Agent Arena (WAA). Given a question / instruction, a multimodal agent runs step-by-step inside a Windows VM; each step’s screenshot and model output are written into one aggregated JSON file.

## Requirements

- Linux host (recommended) with Docker and KVM (`/dev/kvm`)
- OpenAI-compatible API key (or Azure OpenAI)
- Python 3.9+ on the host (for script deps; the agent runs inside the container)
- Optional: local `bert-base-uncased` (needed for GroundingDINO / `som_origin=oss`)

## 1. Build & Setup

### 1.1 Install host dependencies

```bash
pip install -r requirements.txt
```

### 1.2 Configure API keys

Create `config.json` at the repo root of this directory:

```json
{
    "OPENAI_API_KEY": "<YOUR_API_KEY>",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "AZURE_API_KEY": "",
    "AZURE_ENDPOINT": ""
}
```

`OPENAI_API_KEY` + `OPENAI_BASE_URL` take priority; Azure fields are optional alternatives.

### 1.3 Build Docker images

```bash
docker pull windowsarena/winarena-base:latest

cd scripts
./build-container-image.sh
```

You should then have `windowsarena/winarena:latest` locally.

### 1.4 Prepare the Windows 11 golden image (first time only)

1. Download a **Windows 11 Enterprise Evaluation** ISO (~6GB) from the [Microsoft Evaluation Center](https://info.microsoft.com/ww-landing-windows-11-enterprise.html).
2. Rename it to `setup.iso` and place it at:

```text
src/win-arena-container/vm/image/setup.iso
```

3. Run the automated install (~20 minutes; do not interact with the VM):

```bash
cd scripts
./run-local.sh --prepare-image true
```

Monitor progress at `http://localhost:8006`.

When finished, the golden image lives under:

```text
src/win-arena-container/vm/storage/
```

Back up this folder outside the repo so you can recover if the VM is corrupted.

> If your user is not in the `docker` group, use `sg docker -c './run-local.sh ...'` or re-login after joining the group.

## 2. Collect Trajectories

Pipeline:

1. Start the container with the Windows VM (no evaluation / no auto-collect).
2. Run `run_collect.py` inside the container.
3. Get `collection.json` plus per-step screenshots.

### 2.1 Write a questions file

Example: `src/win-arena-container/client/collection_examples/questions.json`

```json
{
  "questions": [
    {
      "id": "open_notepad_hello",
      "instruction": "Please open Notepad and type hello world."
    },
    "Open Calculator and compute 1+1."
  ]
}
```

Supported shapes:

- list of strings: `["q1", "q2"]`
- list of objects: `[{"id": "...", "instruction": "..."}]`
- object with a `questions` field (as above)

For environment setup (e.g. file downloads), add a `config` field using the same schema as original WAA tasks.

### 2.2 Start the Windows environment

```bash
cd scripts
./run-local.sh \
  --skip-build true \
  --start-client false \
  --prepare-image false \
  --container-name winarena
```

Wait until the log shows `VM started, server ready`. Open `http://localhost:8006` to view the desktop.

### 2.3 Run collection inside the container

```bash
docker exec -w /client winarena python run_collect.py \
  --questions_path collection_examples/questions.json \
  --model claude-sonnet-4-5-20250929 \
  --som_origin a11y \
  --a11y_backend uia \
  --max_steps 15 \
  --output_dir ./collection_results
```

Single question:

```bash
docker exec -w /client winarena python run_collect.py \
  --question "Open Notepad and type hello" \
  --question_id demo_hello \
  --model claude-sonnet-4-5-20250929 \
  --som_origin a11y \
  --output_dir ./collection_results
```

Or use the helper script:

```bash
docker exec winarena bash /start_collect.sh \
  --questions-path collection_examples/questions.json \
  --model claude-sonnet-4-5-20250929 \
  --som-origin a11y
```

### 2.4 Common flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--questions_path` | Path to questions JSON | none (use `--question` instead) |
| `--question` | Single question string | none |
| `--output_dir` | Output directory | `./collection_results` |
| `--output_json` | Aggregated JSON path | `<output_dir>/collection.json` |
| `--model` | Model name | `gpt-4-vision-preview` |
| `--som_origin` | Screen parsing: `a11y` / `oss` / … | `oss` |
| `--max_steps` | Max steps per question | `15` |
| `--embed_base64` | Embed screenshots as base64 in JSON | off |
| `--save_user_question` | Also save the prompt sent to the model | off |

Prefer `--som_origin a11y` for collection (more stable). `oss` needs local BERT / GroundingDINO and is heavier.

### 2.5 Output layout

Default output (mounted to the host):

```text
src/win-arena-container/client/collection_results/
├── collection.json
├── screenshots/
│   └── <question_id>/
│       ├── step_0.png
│       ├── step_1.png
│       └── ...
└── logs/
```

`collection.json` sketch:

```json
{
  "created_at": "...",
  "model": "...",
  "episodes": [
    {
      "id": "open_notepad_hello",
      "instruction": "question text",
      "steps": [
        {
          "step": 0,
          "screenshot": "screenshots/open_notepad_hello/step_0.png",
          "model_output": "full model output",
          "action": "parsed action code",
          "done": false
        }
      ],
      "done": true,
      "num_steps": 3
    }
  ]
}
```

Notes:

- Each step stores the screenshot **seen by the model** (pre-action).
- Screenshots are PNG files by default; JSON stores relative paths.
- The JSON is flushed incrementally; finished episodes survive interrupts.

## 3. Desktop-only Mode (install software & persist)

The Windows disk is bind-mounted at `src/win-arena-container/vm/storage/`. If you **shut down Windows cleanly** before stopping the container, installed software persists.

### 3.1 Start desktop (no collection)

```bash
cd scripts
./start-desktop.sh
```

After `VM started, server ready`:

- Browser: `http://localhost:8006`
- RDP: `localhost:3390`

Install apps / change settings like a normal Windows machine.

### 3.2 Save & stop (important)

Do **not** use a raw `docker stop` / `docker kill` (unsynced disk writes may be lost). Use:

```bash
cd scripts
./stop-desktop.sh
```

This will:

1. Call `POST /shutdown` inside the VM for a graceful Windows shutdown  
2. Wait ~3 minutes for the disk flush into `storage/`  
3. Then `docker stop` the container  

### 3.3 Next time

`./start-desktop.sh` or collection runs reuse the same `vm/storage/`, so installed software remains.

Periodically back up `src/win-arena-container/vm/storage/` outside the repo.

## 4. Everyday Commands

```bash
cd scripts

# Desktop only
./start-desktop.sh
./stop-desktop.sh

# Start env, then collect manually
./run-local.sh --skip-build true --start-client false
docker exec -w /client winarena python run_collect.py ...

docker ps | grep winarena
```

Edits under `src/win-arena-container/client/` are bind-mounted and usually apply **without** rebuilding the image. Rebuild with `./build-container-image.sh` only when Dockerfile / system layers change.

## 5. Layout

```text
.
├── config.json                          # API config (gitignored)
├── README.md                            # English (default)
├── README_zh.md                         # Chinese
├── scripts/
│   ├── build-container-image.sh
│   ├── run-local.sh
│   ├── start-desktop.sh
│   └── stop-desktop.sh
└── src/win-arena-container/
    ├── start_collect.sh
    ├── client/
    │   ├── run_collect.py
    │   ├── lib_run_collect.py
    │   ├── collection_recorder.py
    │   ├── collection_examples/
    │   └── collection_results/
    └── vm/
        ├── image/setup.iso
        └── storage/                     # golden image + persisted disk
```

## License

Adapted from Windows Agent Arena under the original [MIT License](LICENSE).
