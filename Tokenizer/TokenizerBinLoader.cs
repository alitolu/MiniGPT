using System.Text;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MiniGPT.Tokenizer
{
    public class TokenizerBinLoader
    {
        public Dictionary<int,string> IdToToken = new();
        Dictionary<string,int> TokenToId = new();

        public void Load(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));

            int vocab = br.ReadInt32();

            for(int i=0;i<vocab;i++)
            {
                int len = br.ReadInt32();
                var bytes = br.ReadBytes(len);
                string token = Encoding.UTF8.GetString(bytes);

                IdToToken[i]=token;
                TokenToId[token]=i;
            }
        }

        public int[] Encode(string text)
            => text.Select(c => TokenToId.ContainsKey(c.ToString()) ? TokenToId[c.ToString()] : 0).ToArray();

        public string Decode(IEnumerable<int> ids)
            => string.Concat(ids.Select(i=>IdToToken.ContainsKey(i) ? IdToToken[i] : ""));
    }
}
