using System.Collections.Generic;
using System.IO;

namespace MiniGPT.Data
{
    public class StreamingDataset
    {
        string path;

        public StreamingDataset(string p)
        {
            path=p;
        }

        public IEnumerable<string> StreamLines()
        {
            using var reader=new StreamReader(path);

            while(!reader.EndOfStream)
                yield return reader.ReadLine();
        }
    }
}
