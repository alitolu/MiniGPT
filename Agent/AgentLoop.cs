using MiniGPT.RAG;
using MiniGPT.Memory;
using MiniGPT.Engine;

namespace MiniGPT.Agent
{
    public class AgentLoop
    {
        ChatEngine engine; 
        ToolRegistry tools;
        Retriever rag;
        ConversationMemory memory;

        public AgentLoop(
            ChatEngine engine,
            ToolRegistry tools,
            Retriever rag,
            ConversationMemory memory)
        {
            this.engine=engine;
            this.tools=tools;
            this.rag=rag;
            this.memory=memory;
        }

        public string Chat(string userInput)
        {
            memory.Add("User: "+userInput);

            // RAG retrieval
            string knowledge = rag.Retrieve(userInput);
            
            string context = 
                "Knowledge:\n" + knowledge + "\n" +
                "History:\n" + memory.Context() + "\n" +
                "User: " + userInput + "\n" +
                "AI:";

            // Generate initial response
            string response = engine.Generate(context);

            // Check for function call
            var call = FunctionCallParser.TryParse(response);

            if(call!=null)
            {
                // Execute tool
                string result = tools.Execute(call);

                // Feed result back to LLM
                string toolContext = context + response + "\nToolResult: " + result + "\nAI:";
                response = engine.Generate(toolContext);
            }

            memory.Add("AI: "+response);
            return response;
        }
    }
}
