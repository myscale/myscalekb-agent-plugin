# MySaleKB Agent Plugin 使用指南

本使用指南重点介绍如何基于 MyScaleKB Agent Docker 运行环境载入 Plugin 代码。

## 如何载入 Plugin

MyScaleKB Agent Plugin(SubAgent) 代码的运行依赖于 MyScaleKB Agent Docker 环境。阅读 [MyScale KB Agent 平台部署指南](https://github.com/myscale/myscalekb-deployment/blob/main/README.md) 以了解如何在本地部署完整的 MyScaleKB Agent 服务。

在此基础上，Plugin 代码使用 Docker Mount 的方式进行载入。

```shell
# 假设目录结构如下
# -- workspace
# ---- myscalekb-agent-plugin
# ---- myscalekb-deployment
cd workspace
git clone https://github.com/myscale/myscalekb-agent-plugin.git

# 以 myscalekb-deployment 为根目录操作 docker-compose
cd myscalekb-deployment

# 使用下面的命令 restart agent service (以 cpu yaml 为例)
docker-compose -f docker-compose-linux-cpu.yaml -v ../myscalekb-agent-plugin/myscalekb_agent_plugin:/app/myscalekb_agent_plugin up -d agent
```

### 查看日志

使用下面的命令来验证 Plugin SubAgent 是否正确载入：
```shell
docker-compose -f docker-compose-linux-cpu.yaml logs agent --since 10m | grep "PaperRecommendationAgent"
```

使用下面的命令来查看 Agent Workflow 的执行情况：
```shell
docker-compose -f docker-compose-linux-cpu.yaml logs agent --since 10m | grep "QueryTrace"
```

也可以通过 TraceId 来定位具体的一次请求执行情况，假设 TraceId 为 `5dd78e08ae994ef898704052b84447b7a373`
```shell
docker-compose -f docker-compose-linux-cpu.yaml logs agent --since 10m | grep -F "QueryTrace[5dd78e08ae994ef898704052b84447b7a373]"
```


## 如何开发 Plugin

MyScaleKB Agent Plugin 的介绍及开发流程请阅读 [MyScaleKB Agent 二次开发用户指南](https://icni9182qqbe.feishu.cn/wiki/V4lwwlSHtilnRqkWybhcMMI8nWb?fromScene=spaceOverview)。
