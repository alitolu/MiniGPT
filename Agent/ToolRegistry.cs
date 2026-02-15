using System.Collections.Generic;

namespace MiniGPT.Agent
{
    public interface ITool
    {
        string Name { get; }
        string Execute(Dictionary<string,object> args);
    }

    public class ToolRegistry
    {
        Dictionary<string,ITool> tools = new();

        public void Register(ITool tool)
            => tools[tool.Name] = tool;

        public string Execute(FunctionCall call)
        {
            if (tools.ContainsKey(call.Tool))
                return tools[call.Tool].Execute(call.Args);
            return "Error: Tool not found";
        }
    }
}
