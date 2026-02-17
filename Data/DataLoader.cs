using MiniGPT.Tokenizer;
using System.Collections.Generic;

namespace MiniGPT.Data
{
    public class DataLoader
    {
        StreamingDataset dataset;
        BPETokenizer tokenizer;
        int batchSize;

        public DataLoader(
            StreamingDataset dataset,
            BPETokenizer tokenizer,
            int batchSize)
        {
            this.dataset = dataset;
            this.tokenizer = tokenizer;
            this.batchSize = batchSize;
        }

        public IEnumerable<List<int[]>> GetBatches()
        {
            var batch = new List<int[]>();

            foreach (var line in dataset.StreamLines())
            {
                // Context Length Limit: 128
                var tokens = tokenizer.Encode(line).Take(128).ToArray();
                batch.Add(tokens);

                if (batch.Count == batchSize)
                {
                    yield return batch;
                    batch = new List<int[]>();
                }
            }

            if (batch.Count > 0)
                yield return batch;
        }
    }
}
