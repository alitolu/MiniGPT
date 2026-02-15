using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Tokenizer
{
    public class BPETokenizer
    {
        Dictionary<string,int> vocab=new();
        Dictionary<int,string> rev=new();

        public int EOS { get; private set; }

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

            vocab["<EOS>"]=id;
            rev[id]="<EOS>";
            EOS=id;
        }

        public int VocabSize => vocab.Count;

        public List<int> Encode(string s)
            => s.Split(' ')
                .Select(w=>vocab.ContainsKey(w)?vocab[w]:0)
                .ToList();

        public string Decode(IEnumerable<int> ids)
            => string.Join(" ",ids.Where(i=>i!=EOS).Select(i=>rev.ContainsKey(i)?rev[i]:"?"));
    }
}
