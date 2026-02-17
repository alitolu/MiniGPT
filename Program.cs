using MiniGPT.Tokenizer;
using MiniGPT.Model;
using MiniGPT.Engine;
using MiniGPT.Server;
using MiniGPT.Agent;
using MiniGPT.Tools;
using MiniGPT.Embeddings;
using MiniGPT.RAG;
using MiniGPT.Memory;
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("MiniGPT v1.2 (Agent)");
        Console.WriteLine("1. Train (Sıfırdan eğit)");
        Console.WriteLine("2. Chat (Console)");
        Console.WriteLine("3. Serve (Web UI + API)");
        Console.WriteLine("4. Agent (Tools + RAG + Memory)");
        Console.Write("Seçim: ");
        
        var choice = Console.ReadLine();
        
        if (choice == "1")
        {
            if(!File.Exists("dataset.txt"))
            {
               File.WriteAllText("dataset.txt", "merhaba dünya bu bir test verisidir yapay zeka öğreniyorum c# ile llm yazıyoruz transformer mimarisi harika function calling nedir rag vector database");
            }
            TrainPipeline.Run("dataset.txt");
        }
        else if (choice == "2" || choice == "3" || choice == "4")
        {
            if (!File.Exists("model_final.ckpt"))
            {
                Console.WriteLine("Hata: model_final.ckpt bulunamadı. Önce 1 ile eğitim yapın.");
                return;
            }

            var tokenizer = new BPETokenizer();
            if(File.Exists("dataset.txt"))
            {
                 var ds = new MiniGPT.Data.StreamingDataset("dataset.txt");
                 tokenizer.Build(string.Join(" ", ds.StreamLines()));
            }
            
            var model = new MiniGPTModel(tokenizer.VocabSize, 128, layers: 4);
            MiniGPT.Engine.CheckpointManager.Load(model, "model_final.ckpt");
            
            // model.Quantize(); // Disabled due to missing Q4 Forward implementation
            model.Mode = MiniGPT.Core.ModelMode.Inference;
            
            var chat = new ChatEngine(tokenizer, model, tokenizer.VocabSize);

            if (choice == "2")
            {
                Console.WriteLine("Console Chat Başladı.");
                while(true)
                {
                    Console.Write("\nSen: ");
                    var input=Console.ReadLine();
                    var reply=chat.Generate(input);
                    Console.WriteLine("MiniGPT: "+reply);
                }
            }
            else if (choice == "3")
            {
                 ApiServer.Run(chat);
            }
            else if (choice == "4")
            {
                 // Agent Mode
                 var tools = new ToolRegistry();
                 tools.Register(new CalculatorTool());
                 
                 var embed = new EmbeddingModel();
                 var store = new VectorStore();
                 // Add some knowledge
                 store.Add(embed.Embed("MiniGPT C# ile yazılmış bir LLM projesidir."), "MiniGPT Projesi");
                 store.Add(embed.Embed("Transformer mimarisi attention mekanizmasına dayanır."), "Transformer");
                 
                 var rag = new Retriever(embed, store);
                 var memory = new ConversationMemory();
                 
                 var agent = new AgentLoop(chat, tools, rag, memory);
                 
                 Console.WriteLine("Agent Başladı (Calculator available).");
                 while(true)
                 {
                    Console.Write("\nUser: ");
                    var input=Console.ReadLine();
                    var reply=agent.Chat(input);
                    Console.WriteLine("Agent: "+reply);
                 }
            }
        }
    }
}
