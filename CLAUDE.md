# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

deepseek-vision 是一个给 DeepSeek（纯文本模型）补齐视觉、联网搜索与网页抓取能力的代理服务。对外同时暴露 Anthropic Messages API 与 OpenAI Chat Completions API，客户端只需一个 Key 即可接入任意 AI 工具。

- 后端：Python 3.12+ / FastAPI，依赖由 `uv` 管理（`pyproject.toml` + `uv.lock`）
- 前端：React 19 + Vite + TypeScript 的配置器界面，构建产物输出到 `app/static/` 并**已提交到仓库**，由 FastAPI 直接托管
- 部署：多阶段 Dockerfile（Node 构建前端 → Python 运行时，非 root 用户运行）

## 常用命令

后端（本地开发）：

```bash
uv sync && uv run python main.py
```

Docker 运行：

```bash
docker build -t deepseek-vision .
docker run -p 8000:8000 deepseek-vision
```

前端（开发时热更新，dev server 把 `/v1`、`/status`、`/health` 代理到 localhost:8000）：

```bash
cd frontend && npm install
cd frontend && npm run dev      # http://localhost:5173
```

前端构建（产物写入 `../app/static`，**修改前端后必须运行并提交构建产物**）：

```bash
cd frontend && npm run build
```

仓库没有测试套件，也没有 linter/格式化配置。

配置通过根目录 `.env` 加载（参考 `.env.example`），至少需要 `ADMIN_PASSWORD`、`MASTER_API_KEY`、`DEEPSEEK_API_KEY`。所有配置项都集中定义在 `app/config.py` 的 `Settings` 类中，新增配置项必须先在这里注册。

## 请求链路（核心架构）

```
客户端 (Anthropic SDK / OpenAI SDK / Claude Code / Cline)
  → /v1/messages 或 /v1/chat/completions
  → vision 中间件（图片块 → 文字描述）
  → web_search / web_fetch 中间件（拦截工具协议 → 代理代执行 → 结果注入上下文）
  → DeepSeek 上游 (Anthropic Messages 兼容接口)
  → 响应回传
```

- `app/main.py`：FastAPI 入口。核心是一个**路由白名单中间件**——只有 `ALLOWED_ROUTES`（公开 API）和 `_ADMIN_ROUTES`（需管理员 token）内的路径才会放行，其余一律 404 并记入扫描日志。**新增端点时必须同步加入白名单**，否则永远打不通。这里还负责安全响应头、CORS、静态文件托管、`/admin/login` 与 `/admin/apply`。
- `app/router.py`：模型路由。模块导入时根据配置构建 backend 并填充 `MODEL_REGISTRY`（客户端可见模型 ID → backend）。`DEEPSEEK_MODELS` 支持 `client-id:upstream-id` 别名语法；加第二个上游参照 `_build_extra_backend`（`EXTRA_BACKEND_*`）。
- `app/backends/`：`LLMBackend` 抽象基类（`invoke` / `stream` / `count_tokens`）+ `MessagesHTTPBackend`（httpx 调用上游 `/v1/messages`）。`_build_body` 里有 DeepSeek 特判：强制 `output_config.effort=max`，并移除 `thinking`、`tool_choice`。

## 协议层

- `app/schemas.py`：Anthropic Messages API 的 Pydantic 模型（各类 content block、请求/响应）。**这是所有中间件的内部统一格式**。
- `app/messages.py`：`POST /v1/messages` + `/v1/messages/count_tokens`。按请求里 tools 的类型分发到视觉/搜索/抓取中间件或直接转发 backend；流式走 `_logged_stream` 统一登记 token 数、耗时、错误。
- `app/openai_compat.py`：`POST /v1/chat/completions`。做 OpenAI ↔ Anthropic 双向格式转换（含流式 delta 转换）。所有 OpenAI 新字段的适配都集中在这个文件。
- `app/auth.py`：`MASTER_API_KEY` 认证（逗号分隔多 Key）。`require_auth` 校验 `x-api-key`；`require_auth_flexible` 额外接受 `Authorization: Bearer <key>`。

## 三个中间件（本项目核心价值）

- `app/vision.py`：扫描消息里的 image 块，用 OpenAI 兼容视觉接口**并行**（`asyncio.gather`）生成描述，替换为 `[Image N] <描述>` 文本块。`VISION_MAX_IMAGES` 限定处理数量，超出部分原样透传。
- `app/web_search.py`：**两轮架构**。第 1 轮让模型一次性规划所有搜索 query → 并行执行（Tavily / Brave）→ 第 2 轮基于结果生成最终答案。用两轮替代多轮循环，避免 O(n²) token 膨胀。
- `app/web_fetch.py`：**agentic 循环**（最多 25 轮，直到模型不再请求抓取或 `max_uses` 耗尽）。带 SSRF 防护与 DNS pinning（自定义 httpcore network backend）；支持 HTML/纯文本/PDF，有 `allowed_domains` / `blocked_domains` 限制。
- 服务端工具协议技巧：上游不支持 Anthropic 的 server-side 工具，代理拦截 `web_search_*` / `web_fetch_*` 工具标记、自己执行，再把 `tool_use` 转成 `server_tool_use` 块回传给客户端。`_to_server_id`、`_convert_to_server_tool_use`、`_post_process_citations`（把 `[N]` 标记转成 citation 对象）是两者共用的同一套模式，**改一处记得检查另一处**。
- 流式是"混合式"：web_search / web_fetch 内部走非流式 agentic 循环，外层 `stream_web_search` / `stream_web_fetch` 起一个后台任务，等待期间持续发 ping 心跳，结束后再把全部 content 块以 SSE 重放。

## 管理端

- `app/admin_auth.py`：管理员 token = `HMAC-SHA256(ADMIN_PASSWORD, 时间戳)`，24 小时过期；登录每 IP 5 分钟限 5 次。
- `app/main.py` 里的 `/admin/apply`：把前端提交的 `.env` 文本原子写盘（临时文件 + rename），随后 SIGTERM 触发自身重启。

## 前端

`frontend/src/App.tsx` 单文件实现配置器：登录页 + 状态卡片 + `.env` 生成器。它轮询 `/status` 展示后端运行状态，把表单序列化为 `.env` 文本 POST 给 `/admin/apply`。样式在 `frontend/src/index.css`，UI 文案为中文。

## 调试

- `LOG_LEVEL=DEBUG` 或 `DEBUG_UPSTREAM=true`（后者会把上游请求体打到日志）。
- 请求统计日志统一以 `[REQ]` 前缀输出到 `logs/app.log`（每日轮转）；`TRACEMALLOC_FRAMES` 可开启 tracemalloc 诊断。