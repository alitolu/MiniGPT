using System.Text.Json;
using System.Collections.Generic;
using System;

namespace MiniGPT.Agent
{
    public record FunctionCall(string Tool, Dictionary<string,object> Args);

    public static class FunctionCallParser
    {
        public static FunctionCall TryParse(string text)
        {
            try
            {
                // Simple heuristic: check if text contains JSON-like structure
                int start = text.IndexOf("{");
                int end = text.LastIndexOf("}");
                
                if (start == -1 || end == -1 || end <= start) return null;

                var json = text.Substring(start, end - start + 1);
                var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                if (root.TryGetProperty("tool", out var toolProp))
                {
                    string tool = toolProp.GetString();
                    var args = new Dictionary<string, object>();

                    if (root.TryGetProperty("args", out var argsProp))
                    {
                        foreach (var prop in argsProp.EnumerateObject())
                        {
                            // Basic type handling
                            if (prop.Value.ValueKind == JsonValueKind.Number)
                                args[prop.Name] = prop.Value.GetDouble();
                            else if (prop.Value.ValueKind == JsonValueKind.String)
                                args[prop.Name] = prop.Value.GetString();
                            else if (prop.Value.ValueKind == JsonValueKind.True)
                                args[prop.Name] = true;
                            else if (prop.Value.ValueKind == JsonValueKind.False)
                                args[prop.Name] = false;
                        }
                    }
                    
                    return new FunctionCall(tool, args);
                }
                return null;
            }
            catch
            {
                return null;
            }
        }
    }
}
