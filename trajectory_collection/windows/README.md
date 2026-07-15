# Windows Agent Arena — 轨迹采集

基于 Windows Agent Arena（WAA）改造的 **GUI Agent 轨迹采集** 仓库：输入问题，让多模态 Agent 在 Windows 虚拟机中逐步执行，并把每一步的截图与模型输出整理进同一个 JSON 文件。

## 环境要求

- Linux 主机（推荐），已安装并可使用 Docker（需支持 KVM：`/dev/kvm`）
- OpenAI 兼容 API Key（或 Azure OpenAI）
- Python 3.9+（主机侧仅用于跑脚本依赖；Agent 实际在容器内运行）
- 可选：本地 `bert-base-uncased` 模型目录（离线跑 GroundingDINO / `som_origin=oss` 时需要）

## 一、构建与初始化

### 1. 安装主机依赖

```bash
cd WindowsAgentArena-main
pip install -r requirements.txt
```

### 2. 配置 API Key

在仓库根目录创建或编辑 `config.json`：

```json
{
    "OPENAI_API_KEY": "<你的 API Key>",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "AZURE_API_KEY": "",
    "AZURE_ENDPOINT": ""
}
```

优先使用 `OPENAI_API_KEY` + `OPENAI_BASE_URL`；也可改用 Azure 字段。

### 3. 准备 Docker 镜像

```bash
# 拉取基础镜像
docker pull windowsarena/winarena-base:latest

# 构建本仓库的 winarena 镜像（包含 src 代码）
cd scripts
./build-container-image.sh
```

构建完成后本地应有镜像：`windowsarena/winarena:latest`。

### 4. 准备 Windows 11 金镜像（首次必须）

1. 从 [Microsoft Evaluation Center](https://info.microsoft.com/ww-landing-windows-11-enterprise.html) 下载 **Windows 11 Enterprise Evaluation** ISO（约 6GB）
2. 重命名为 `setup.iso`，放到：

```text
src/win-arena-container/vm/image/setup.iso
```

3. 启动自动安装（约 20 分钟，过程中不要手动操作 VM）：

```bash
cd scripts
./run-local.sh --prepare-image true
```

进度可在浏览器打开：`http://localhost:8006`。

完成后，金镜像文件会出现在：

```text
src/win-arena-container/vm/storage/
```

建议把该目录备份到仓库外，避免以后被误改后重装。

> 若本机用户不在 `docker` 组，可用 `sg docker -c './run-local.sh ...'`，或把用户加入 docker 组后重新登录。

## 二、采集轨迹

整体流程：

1. 启动带 Windows VM 的容器（不自动跑评测）
2. 在容器内执行 `run_collect.py`
3. 得到 `collection.json` + 逐步截图

### 1. 编写问题文件

示例：`src/win-arena-container/client/collection_examples/questions.json`

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

支持的格式：

- 字符串列表：`["问题1", "问题2"]`
- 对象列表：`[{"id": "...", "instruction": "..."}]`
- 带 `questions` 字段的对象（如上）

如需下载文件等环境初始化，可在问题对象里加 `config`（写法与原 WAA 任务配置相同）。

### 2. 启动 Windows 环境

```bash
cd scripts
./run-local.sh \
  --skip-build true \
  --start-client false \
  --prepare-image false \
  --container-name winarena
```

等待日志出现 `VM started, server ready`。可用浏览器 `http://localhost:8006` 查看桌面。

### 3. 在容器内开始采集

```bash
docker exec -w /client winarena python run_collect.py \
  --questions_path collection_examples/questions.json \
  --model claude-sonnet-4-5-20250929 \
  --som_origin a11y \
  --a11y_backend uia \
  --max_steps 15 \
  --output_dir ./collection_results
```

单个问题也可以：

```bash
docker exec -w /client winarena python run_collect.py \
  --question "Open Notepad and type hello" \
  --question_id demo_hello \
  --model claude-sonnet-4-5-20250929 \
  --som_origin a11y \
  --output_dir ./collection_results
```

或使用封装脚本（容器内）：

```bash
docker exec winarena bash /start_collect.sh \
  --questions-path collection_examples/questions.json \
  --model claude-sonnet-4-5-20250929 \
  --som-origin a11y
```

### 4. 常用参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `--questions_path` | 问题 JSON 路径 | 无（与 `--question` 二选一） |
| `--question` | 单个问题字符串 | 无 |
| `--output_dir` | 输出目录 | `./collection_results` |
| `--output_json` | 汇总 JSON 路径 | `<output_dir>/collection.json` |
| `--model` | 模型名 | `gpt-4-vision-preview` |
| `--som_origin` | 屏幕解析来源：`a11y` / `oss` / … | `oss` |
| `--max_steps` | 每个问题最大步数 | `15` |
| `--embed_base64` | 截图以 base64 写入 JSON | 关闭 |
| `--save_user_question` | 额外保存发给模型的 prompt | 关闭 |

推荐采集时用 `--som_origin a11y`（更稳）；`oss` 依赖本地 BERT / GroundingDINO，更重。

### 5. 输出结果

默认输出目录（挂载到宿主机）：

```text
src/win-arena-container/client/collection_results/
├── collection.json          # 所有问题的轨迹汇总
├── screenshots/
│   └── <question_id>/
│       ├── step_0.png
│       ├── step_1.png
│       └── ...
└── logs/
```

`collection.json` 结构概要：

```json
{
  "created_at": "...",
  "model": "...",
  "episodes": [
    {
      "id": "open_notepad_hello",
      "instruction": "问题文本",
      "steps": [
        {
          "step": 0,
          "screenshot": "screenshots/open_notepad_hello/step_0.png",
          "model_output": "模型完整输出",
          "action": "解析出的动作代码",
          "done": false
        }
      ],
      "done": true,
      "num_steps": 3
    }
  ]
}
```

说明：

- 每一步记录的是**模型决策时看到的截图**（动作执行前）
- 截图默认存 PNG，JSON 里写相对路径
- 采集过程增量写盘，中断后已完成的 episode 仍会保留

## 三、只开桌面（安装软件并持久化）

Windows 磁盘挂载在宿主机目录 `src/win-arena-container/vm/storage/`。只要**正常关机**再停容器，装过的软件下次启动还在。

### 1. 启动桌面（不采集）

```bash
cd scripts
./start-desktop.sh
```

等日志出现 `VM started, server ready` 后：

- 浏览器打开：`http://localhost:8006`
- 或用 RDP 连接：`localhost:3390`

在里面像普通 Windows 一样安装软件、改设置。

### 2. 保存并关闭（重要）

**不要**直接 `docker stop` / `docker kill`（可能丢未落盘改动）。用：

```bash
cd scripts
./stop-desktop.sh
```

脚本会：

1. 调用 VM 内 `POST /shutdown` 让 Windows 正常关机  
2. 等待约 3 分钟把改写刷进 `storage/`  
3. 再 `docker stop` 容器  

### 3. 下次再用

再次 `./start-desktop.sh` 或跑采集，都会加载同一份 `vm/storage/`，之前装的软件还在。

建议定期把整个 `src/win-arena-container/vm/storage/` 备份到仓库外。

## 四、日常常用命令

```bash
cd scripts

# 只开桌面装软件
./start-desktop.sh
./stop-desktop.sh

# 开环境后自己进容器采轨迹
./run-local.sh --skip-build true --start-client false
docker exec -w /client winarena python run_collect.py ...

# 查看容器
docker ps | grep winarena
```

代码改动在 `src/win-arena-container/client/` 下时，因该目录已挂载进容器，一般**不用重建镜像**即可生效；改动容器系统层或 Dockerfile 时再执行 `./build-container-image.sh`。

## 五、目录说明

```text
WindowsAgentArena-main/
├── config.json                          # API 配置
├── scripts/
│   ├── build-container-image.sh         # 构建镜像
│   ├── run-local.sh                     # 启动/准备环境
│   ├── start-desktop.sh                 # 只开桌面（装软件）
│   └── stop-desktop.sh                  # 正常关机并落盘
└── src/win-arena-container/
    ├── start_collect.sh                 # 容器内采集入口
    ├── client/
    │   ├── run_collect.py               # 采集主程序
    │   ├── lib_run_collect.py
    │   ├── collection_recorder.py
    │   ├── collection_examples/         # 示例问题
    │   └── collection_results/          # 采集输出（运行后生成）
    └── vm/
        ├── image/setup.iso              # Windows 安装盘（需自行放入）
        └── storage/                     # 金镜像 + 持久化磁盘（软件装这里）
```

## License

本仓库基于原 Windows Agent Arena 项目改造，遵循原项目 [MIT License](LICENSE)。
