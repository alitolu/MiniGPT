using MiniGPT.Agent;
using System;
using System.Collections.Generic;

namespace MiniGPT.Tools
{
    public class CalculatorTool : ITool
    {
        public string Name => "calculator";

        public string Execute(Dictionary<string,object> args)
        {
            if (!args.ContainsKey("a") || !args.ContainsKey("b")) return "Error: Missing arguments a or b";
            
            double a = Convert.ToDouble(args["a"]);
            double b = Convert.ToDouble(args["b"]);

            // Support operation? Default sum for now or strictly defined
            // Proposal said "a+b"
            return (a + b).ToString();
        }
    }
}
