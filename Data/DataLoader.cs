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
                var tokens = tokenizer.Encode(line);
                batch.Add(tokens.ToArray());

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
