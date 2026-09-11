# cc-switch

在同一台机器上切换 Codex agent，避免把不同供应商的 API key 混在一起。

这是一个本地命令行工具。它给每个 agent 保存一份 toml 和一份密钥，再把当前选中的 agent 拷进 `~/.codex/config.toml`。

默认三个 agent：

- `codex`：自定义供应商，密钥走 `OPENAI_API_KEY` / `auth.json`
- `wenle`：`https://api.wenle.ai/v1`，每次请求从本地密钥文件取 token
- `openai`：官方 ChatGPT 登录，不使用 API key 文件

## 为什么 wenle 必须用文件取 token

第三方供应商不要写 `requires_openai_auth = true`。

这个开关会让 Codex 忽略供应商自己的 key，继续用进程里的 OpenAI 凭证。结果是 toml 已经切到 wenle，但请求仍带着旧的 `OPENAI_API_KEY`。服务端会返回：

```text
POST https://api.wenle.ai/v1/responses
503 没有可用token
```

请求确实打到了 wenle，Authorization 头也在。带上去的是上一套供应商的 key，不是 wenle 的 key。

所以 wenle 用 command auth：

```toml
[model_providers.wenle.auth]
command = "/abs/path/to/cc-switch"
args = ["token", "wenle"]
timeout_ms = 5000
refresh_interval_ms = 0
```

`command` 会写成当前机器上 `cc-switch` 的绝对路径。Codex 每次请求调用 `cc-switch token wenle` 取 bearer token，OpenAI 登录保持不动。

## 目录

```text
~/.local/bin/cc-switch
~/.codex/config.toml                  # 正在使用的配置
~/.codex/auth.json                    # OpenAI / Codex 登录态
~/.codex/cc-switch/current            # 当前 agent 名
~/.codex/cc-switch/agents/<name>.toml
~/.codex/cc-switch/secrets/<name>.key
~/.codex/cc-switch/backups/
```

密钥文件权限 `0600`，密钥目录权限 `0700`。这些文件不要提交到 git。

## 安装

需要 Python 3.10+，并且 `codex` 已在 `PATH` 里。

```bash
install -m 755 cc-switch ~/.local/bin/cc-switch
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cc-switch help
```

第一次运行会创建 `~/.codex/cc-switch/`，把当前 live 配置保存为 agent `codex`，并补上 `wenle` 和 `openai`。

切换到 wenle 前，先把 key 写进去：

```bash
umask 077
printf '%s\n' 'sk-...' > ~/.codex/cc-switch/secrets/wenle.key
chmod 600 ~/.codex/cc-switch/secrets/wenle.key
```

不要把 key 写进 toml。

## 命令

```bash
cc-switch help
cc-switch switch <name>
cc-switch <name>
cc-switch open
cc-switch add <name>
cc-switch logout
cc-switch token <name>
```

`help` 打印当前 agent、鉴权类型、key 状态、模型、供应商和 base URL。

`switch` 会先把当前 live `config.toml` 写回当前 agent，备份 live 的 `config.toml` / `auth.json`，再装上目标 agent。

- `codex`：把 key 写入 `~/.bashrc`、`~/.profile`、`auth.json` 的 `OPENAI_API_KEY`，然后执行 `codex login --with-api-key`
- `wenle`：只拷 toml，不改 OpenAI 登录
- `openai`：清掉 `OPENAI_API_KEY`，执行 `codex login --device-auth`

`open` 用 `nano` 编辑当前 agent 的 toml，再拷回 live 配置。

`add <name>` 从当前 live 配置克隆一个新 agent。名字必须匹配 `[A-Za-z0-9][A-Za-z0-9_-]*`。保留名：`help`、`switch`、`open`、`add`、`logout`、`token`。

`logout` 清掉当前 agent 的 live 凭证。文件里的密钥会保留。

`token <name>` 打印密钥。Codex 用它给 wenle 取 token。

切换后：

```bash
source ~/.bashrc
```

然后完全退出并重新打开 Codex。正在跑的进程会继续用旧的 provider 和 key。

## help 示例

```text
cc-switch: 3 agent(s), current=codex

commands:
  cc-switch help
  cc-switch switch <name>
  cc-switch open
  cc-switch add <name>
  cc-switch logout
  cc-switch token <name>

agents:
 * codex        apikey   key ready/OPENAI_API_KEY model=gpt-5.6-sol provider=codex url=https://code.codingplay.top
   wenle        apikey   key ready/file model=gpt-5.6-sol provider=wenle url=https://api.wenle.ai/v1
   openai       chatgpt  account login  model=gpt-5.6-luna provider=openai url=-
```

## 排错

wenle 返回 503 / `没有可用token` 时：

1. 确认 live 配置已经是 wenle。
2. 确认密钥文件非空，并且 `cc-switch token wenle` 能打印出 key。
3. 完全重启 Codex，不要复用旧会话。
4. 如果 toml 里还有 `requires_openai_auth = true`，先跑一次 `cc-switch help`。脚本会把 wenle 改回 command auth。

切到 `openai` 时如果停在 device login，这是预期行为，把设备码走完即可。

`cc-switch open` 依赖 `nano`。

## License

MIT
