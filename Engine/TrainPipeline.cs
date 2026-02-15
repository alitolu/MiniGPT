using MiniGPT.Core;
using MiniGPT.Data;
using MiniGPT.Tokenizer;
using MiniGPT.Model;
using System;

namespace MiniGPT.Engine
{
    public class TrainPipeline
    {
        public static void Run(string datasetPath)
        {
            var tokenizer = new BPETokenizer();

            var dataset = new StreamingDataset(datasetPath);

            var corpus = string.Join(" ", dataset.StreamLines());
            tokenizer.Build(corpus);

            var model = new MiniGPTModel(
                tokenizer.VocabSize, 128, layers: 4);

            model.Mode = ModelMode.Train;

            var loader = new DataLoader(dataset, tokenizer, 8);

            var trainer = new Trainer(model, tokenizer.VocabSize);

            int epoch = 0;

            foreach (var batch in loader.GetBatches())
            {
                trainer.TrainBatch(batch);

                if (epoch % 10 == 0)
                    CheckpointManager.Save(
                        model,
                        $"model_epoch_{epoch}.ckpt");

                epoch++;
            }

            model.Mode = ModelMode.Inference;
            model.DisableGradients();
            model.Quantize();

            CheckpointManager.Save(model, "model_final.ckpt");

            Console.WriteLine($"Training tamamlandı. {epoch} batch işlendi.");
        }
    }
}
