using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Tokenizer
{
    public class BPETokenizer
    {
        Dictionary<string,int> vocab=new();
        Dictionary<int,string> rev=new();

        public void Build(string text)
        {
            var words=text.Split(' ').Distinct();
            int id=0;

            foreach(var w in words)
            {
                vocab[w]=id;
                rev[id]=w;
                id++;
            }
        }

        public int VocabSize => vocab.Count;

        public int[] Encode(string s)
            => s.Split(' ')
                .Select(w=>vocab.ContainsKey(w)?vocab[w]:0)
                .ToArray();

        public string Decode(IEnumerable<int> ids)
            => string.Join(" ",ids.Select(i=>rev[i]));
    }
}
