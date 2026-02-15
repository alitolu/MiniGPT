using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using MiniGPT.Engine;
using MiniGPT.Tokenizer;

namespace MiniGPT.Runtime
{
    public class StreamingGenerator
    {
        ChatEngine engine; // Reuse ChatEngine for stateful generation

        public StreamingGenerator(ChatEngine e)
        {
            engine = e;
        }

        public async IAsyncEnumerable<string> GenerateStream(
            string prompt,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            // ChatEngine doesn't support streaming (yield return) yet.
            // We need to modify ChatEngine or implement streaming here.
            // Let's modify ChatEngine to be enumerable? Or just simulate here.
            
            // Simulation for Phase-11 demonstration:
            var fullReply = engine.Generate(prompt, maxTokens);
            var words = fullReply.Split(' ');
            
            foreach (var w in words)
            {
                yield return w + " ";
                await Task.Delay(50, ct); // Simulate thinking
            }
        }
    }
}
