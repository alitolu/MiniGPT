using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Tokenizer
{
    public class BPETokenizerTrainer
    {
        public Dictionary<string,int> Train(
            IEnumerable<string> corpus,
            int vocabSize)
        {
            var vocab = new Dictionary<string,int>();

            var words = corpus
                .SelectMany(x => x.Split(' '))
                .ToList();

            var tokens = words
                .Select(w => string.Join(" ", w.ToCharArray()))
                .ToList();

            while (vocab.Count < vocabSize)
            {
                var pairs = new Dictionary<string,int>();

                foreach (var t in tokens)
                {
                    var parts = t.Split(' ');
                    for(int i=0;i<parts.Length-1;i++)
                    {
                        var pair = parts[i]+" "+parts[i+1];
                        pairs[pair] = pairs.GetValueOrDefault(pair)+1;
                    }
                }

                if (pairs.Count == 0) break;

                var best = pairs
                    .OrderByDescending(x=>x.Value)
                    .First().Key;

                vocab[best] = vocab.Count;

                tokens = tokens
                    .Select(t => t.Replace(best, best.Replace(" ","")))
                    .ToList();
            }

            return vocab;
        }
    }
}
