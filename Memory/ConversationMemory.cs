using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Memory
{
    public class ConversationMemory
    {
        List<string> history = new();

        public void Add(string msg)
            => history.Add(msg);

        public string Context(int last=6)
        {
            return string.Join("\n",
                history.TakeLast(last));
        }
    }
}
