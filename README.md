# deepseek-vision

**为 DeepSeek 补齐视觉理解、联网搜索与 Anthropic / OpenAI 兼容接口的代理服务。**

[English](./README.en.md)

DeepSeek 官方 API 是纯文本模型，这会极大限制 Agent 的能力以及用户的对话体验，尤其是在 Claude Code 等场景使用时还需要额外的联网搜索和抓取能力，本项目也一并补齐。

本项目是 [Proma](https://proma.cool) 的旁支开源项目。Proma 是最丝滑的通用开源 Agent，对 DeepSeek v4 系列的适配最为完整，已在云端服务中补齐了包括视觉、联网搜索在内的全部缺失能力，欢迎直接使用。本仓库提供可自部署的代理版本，让你用同一个 DeepSeek API Key 接入任何 AI 工具。

> **上游来源**：本仓库 fork 自 [ErlichLiu/deepseek-vision](https://github.com/ErlichLiu/deepseek-vision) 并二次开发，在此基础上新增了 OpenAI 格式上游（`DEEPSEEK_FORMAT` / `EXTRA_BACKEND_FORMAT`）、强化视觉中间件（思考模型适配、工具结果内嵌图片、失败重试）、思考过程透传与多轮 `reasoning_content` 回传，并修复了 Windows 下前端白屏等问题。代码以 MIT 协议发布，保留原版权声明，感谢原作者与开源社区。

---

## 快速开始

### 配置器（推荐）

启动后访问 `http://localhost:8000`，在配置器页面填写 API Key，点击「应用并重启」即可。

```bash
# Docker
docker build -t deepseek-vision .
docker run -p 8000:8000 deepseek-vision
```

### 使用预编译镜像（推荐，无需本地构建）

镜像由 GitHub Actions 自动发布到 GitHub Container Registry，支持 `linux/amd64` 与 `linux/arm64`：

```bash
docker pull ghcr.io/dijiaozhibei-top/deepseek-vision:latest
docker run -p 8000:8000 --env-file .env ghcr.io/dijiaozhibei-top/deepseek-vision:latest

# 或使用仓库内的 docker-compose.yml（自动读取 .env、端口可用 ${PORT} 自定义）
docker compose up -d
```

预编译二进制（Windows / Linux / macOS，x64 / arm64）可在 [Releases](https://github.com/dijiaozhibei-top/deepseek-vision/releases) 页面下载。

### 手动配置

```bash
cp .env.example .env
# 编辑 .env，至少填写：ADMIN_PASSWORD、MASTER_API_KEY、DEEPSEEK_API_KEY
```

```bash
# Docker
docker run --env-file .env -p 8000:8000 deepseek-vision

# 本地（uv）
uv sync && uv run python main.py

# 本地（pip）
pip install . && python main.py
```

---

## 界面预览

<table>
  <tr>
    <td align="center"><b>登录页</b></td>
    <td align="center"><b>配置器</b></td>
  </tr>
  <tr>
    <td><img src="./docs/screenshot-login.png" alt="登录页" width="380"/></td>
    <td><img src="./docs/screenshot-dashboard.png" alt="配置器" width="380"/></td>
  </tr>
</table>

---

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/messages` | Anthropic Messages API |
| `POST` | `/v1/messages/count_tokens` | Token 计数 |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions API |
| `GET`  | `/v1/models` | 查询可用模型 |
| `GET`  | `/health` | 存活检查 |
| `GET`  | `/` | 配置器 UI |

所有 API 端点需要通过 `x-api-key` 请求头（Anthropic 风格）或 `Authorization: Bearer <key>`（OpenAI 风格）传入 `MASTER_API_KEY`。

---

## 工作原理

```
客户端（Anthropic SDK / OpenAI SDK / LangChain / Cline）
    │
    ├─ POST /v1/chat/completions  ──►  OpenAI → Anthropic 格式转换
    │                                            │
    └─ POST /v1/messages  ──────────────────────►┤
                                                 │
                                       视觉中间件
                                       图片块 → 调用视觉模型 → 文字描述
                                                 │
                                       web_search / web_fetch 中间件
                                       Anthropic 工具协议 → Tavily/Brave
                                       → 结果注入上下文
                                                 │
                                       DeepSeek 上游
                                       （Anthropic Messages API）
                                                 │
                                       响应 → 返回给客户端
```

---

## 视觉补齐

默认使用阿里云 Qwen（`qwen3.6-flash`），只需填写 `VISION_API_KEY` 即可启用：

```env
VISION_API_KEY=sk-your-dashscope-key
```

每个请求里的 `image` 内容块会被替换为 `[Image N] <描述文字>` 的文本块，多张图片并行处理。

也可以替换为其他 OpenAI 兼容的视觉模型：

```env
VISION_BASE_URL=https://api.openai.com/v1
VISION_API_KEY=sk-...
VISION_MODEL=gpt-4o-mini
```

支持的视觉后端（任何具有 OpenAI 兼容接口的服务）：
- 阿里云 DashScope（`qwen3.6-flash`、`qwen-vl-max` 等）
- OpenAI（`gpt-4o`、`gpt-4o-mini`）
- GLM-4V、InternVL、LLaVA（通过 vLLM 自部署）

---

## 联网搜索与网页抓取

在请求中使用 Anthropic 工具协议添加 `web_search` 或 `web_fetch` 工具。代理会拦截工具调用、执行搜索/抓取，并将结果注回上下文——DeepSeek 本身仍只做文本生成。

### web_search

两轮架构：第一轮让模型规划所有查询（并行执行），第二轮基于搜索结果生成最终答案。结果自动附加 `[N]` 引用标注。

配置 Tavily（推荐）或 Brave：

```env
TAVILY_API_KEY=tvly-...
# 或
WEB_SEARCH_PROVIDER=brave
BRAVE_API_KEY=BSA-...
```

### web_fetch

带 SSRF 防护和 DNS pinning 的 URL 抓取。支持 HTML、纯文本和 PDF。结果自动附加 `[Document N]` 引用标注。无需额外配置。

---

## 模型配置

默认暴露 `deepseek-v4-pro` 和 `deepseek-v4-flash`，可通过 `DEEPSEEK_MODELS` 自定义：

```env
# 直接使用上游 ID
DEEPSEEK_MODELS=deepseek-v4-pro,deepseek-v4-flash

# 使用别名（client-id:upstream-id）
DEEPSEEK_MODELS=pro:deepseek-v4-pro,flash:deepseek-v4-flash
```

通过 `EXTRA_BACKEND_*` 添加第二个 Anthropic 兼容上游：

```env
EXTRA_BACKEND_NAME=my-provider
EXTRA_BACKEND_BASE_URL=https://api.example.com/anthropic
EXTRA_BACKEND_API_KEY=sk-...
EXTRA_BACKEND_MODELS=model-a,model-b
```

上游接口默认走 Anthropic Messages 格式。若供应商只提供 OpenAI Chat Completions 接口（如 DeepSeek 官方 OpenAI 端点、vLLM、Ollama），把 `DEEPSEEK_FORMAT`（或额外上游的 `EXTRA_BACKEND_FORMAT`）设为 `openai`，并把对应 `BASE_URL` 换成 OpenAI 风格根地址：

```env
DEEPSEEK_FORMAT=openai
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

---

## 配置项说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ADMIN_PASSWORD` | `123456` | 配置器登录密码，**请修改** |
| `MASTER_API_KEY` | 必填 | 客户端访问代理所用的 Key，逗号分隔支持多个 |
| `DEEPSEEK_API_KEY` | 必填 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/anthropic` | DeepSeek 上游地址 |
| `DEEPSEEK_FORMAT` | `anthropic` | 上游接口格式：`anthropic`（Messages API，默认）或 `openai`（Chat Completions）。切到 `openai` 时 `DEEPSEEK_BASE_URL` 应填 OpenAI 风格根地址（如 `https://api.deepseek.com`） |
| `EXTRA_BACKEND_FORMAT` | `anthropic` | 仅用于额外上游（`EXTRA_BACKEND_*`），取值与说明同上 |
| `DEEPSEEK_MODELS` | `deepseek-v4-pro,deepseek-v4-flash` | 暴露给客户端的模型列表 |
| `VISION_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 视觉模型接口地址 |
| `VISION_API_KEY` | — | 视觉模型 API Key（留空则禁用视觉补齐） |
| `VISION_MODEL` | `qwen3.6-flash` | 视觉模型名称 |
| `VISION_MAX_IMAGES` | `5` | 单次请求最多处理的图片数量 |
| `WEB_SEARCH_PROVIDER` | `tavily` | 搜索服务商（`tavily` 或 `brave`） |
| `TAVILY_API_KEY` | — | Tavily API Key |
| `BRAVE_API_KEY` | — | Brave Search API Key |
| `PORT` | `8000` | 服务端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

---

## Roadmap

- [ ] `/v1/embeddings` 接口
- [ ] SearXNG 搜索支持（自部署）
- [ ] OpenAI 兼容模式下的流式工具调用

---

## License

MIT
