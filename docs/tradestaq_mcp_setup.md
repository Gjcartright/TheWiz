# TradeStaq MCP Setup

Use this MCP server config in the MCP client you are running, for example Claude Desktop, Cursor, or another MCP-compatible host:

```json
{
  "mcpServers": {
    "tradestaq": {
      "command": "npx",
      "args": ["-y", "@the-staq/tradestaq-mcp"]
    }
  }
}
```

A repo-local copy is stored at:

```text
mcp.example.json
```

Notes:

- This file does not contain secrets.
- The MCP host must have Node.js and `npx` available.
- The first run may download `@the-staq/tradestaq-mcp`.
- If your MCP host uses a different schema, copy only the inner server entry and adapt the wrapper key.
