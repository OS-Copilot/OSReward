# Judge prompts

System prompts for the judge, plain UTF-8. Pick one with `--prompt <file>`
(default `multi_v4.txt`); it is loaded verbatim by `config.load_system_prompt`.

Two output schemas:
- `multi_*` — asks for `Thought` + `Judge` + `Alignment` + `Efficiency`.
- `binary_*` — asks for `Thought` + `Judge` only.

This package scores **only the binary `Judge` (SUCCESS/FAIL)**; the extra
`Alignment`/`Efficiency` fields a `multi_*` prompt emits are simply ignored here.

| File | Schema | Notes |
|---|---|---|
| `multi_v4.txt` | multi | **default**. Single-shot Alignment + Efficiency rubric; binary section identical to `v1.txt`. |
| `multi_v4_minimal.txt` | multi | alias of `multi_v4.txt` (kept so older runs reproduce). |
| `multi_v4.1.txt` | multi | experimental variant (tighter Alignment, looser Efficiency). |
| `multi_v3.1.txt` | multi | superseded; uses an Alignment Q1/Q2 decomposition. |
| `binary_v1.txt` | binary | minimal `Thought` + `Judge` prompt. |
| `v1.txt` / `v2.txt` / `v3.txt` / `v3.1.txt` | multi (legacy) | earlier iterations, kept for reproducibility. |

To add a prompt: write `<schema>_<version>.txt` whose output lines match the
parser (`Judge:` for binary; plus `Alignment:`/`Efficiency:` for multi), then
select it with `--prompt`.
