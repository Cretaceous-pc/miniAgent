# 文档服务接口约定

## 鉴权

所有请求需带 `Authorization: Bearer <token>`。
token 有效期为 2 小时，过期返回 `40101`。

## 接口列表

### POST /v1/chat/completions

请求体：

```json
{
  "model": "demo-chat",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0.3
}
```

响应体：

```json
{
  "id": "chatcmpl-1",
  "choices": [{"message": {"role": "assistant", "content": "你好"}}]
}
```

### POST /v1/documents/search

请求体：`{"query": "报销", "top_k": 5}`
响应体：`{"hits": [{"path": "handbook.md", "line": 27, "score": 0.82}]}`

## 错误码

| 错误码 | 含义 | 是否可重试 |
| --- | --- | --- |
| 40101 | token 过期 | 否，需重新鉴权 |
| 42901 | 触发限流 | 是，建议退避 1s / 2s |
| 50001 | 文档解析失败 | 否 |
| 50002 | 向量库超时 | 是 |

## 已知问题

- TODO：`/v1/documents/search` 暂不支持按目录过滤。
- TODO：错误码还需要补充客户端侧的映射表。
