using MiniGPT.Core;
using MiniGPT.Data;
using MiniGPT.Tokenizer;
using MiniGPT.Model;
using System;
using System.IO;
using System.Linq;

namespace MiniGPT.Engine
{
    public class TrainPipeline
    {
        public static void Run(string datasetPath)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("==================================================");
            Console.WriteLine("MiniGPT Training System v1.2");
            Console.WriteLine($"OS: {Environment.OSVersion}");
            Console.WriteLine($"CPU Cores: {Environment.ProcessorCount}");
            Console.WriteLine($"Memory Usage: {System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / 1024 / 1024} MB");
            Console.WriteLine("GPU Status: N/A (Running on CPU Only Mode)");
            Console.WriteLine("==================================================");
            Console.WriteLine("Dataset yükleniyor...");

            var tokenizer = new BPETokenizer();

            var dataset = new StreamingDataset(datasetPath);

            var corpus = string.Join(" ", dataset.StreamLines());
            tokenizer.Build(corpus);

            var model = new MiniGPTModel(
                tokenizer.VocabSize, 128, layers: 4);

            model.Mode = ModelMode.Train;

            var batchSize = 8;
            var loader = new DataLoader(dataset, tokenizer, batchSize);
            var trainer = new Trainer(model, tokenizer.VocabSize);
            
            // Satır sayısını hesapla
            var totalLines = File.ReadLines(datasetPath).Count();
            var totalBatches = (int)Math.Ceiling((double)totalLines / batchSize);

            for (int epoch = 1; epoch <= 50; epoch++)
            {
                Console.WriteLine($"\nEpoch {epoch}/50 Başlıyor... (Toplam {totalBatches} Batch)");
                var currentBatch = 0;
                
                foreach (var batch in loader.GetBatches())
                {
                    trainer.TrainBatch(batch);
                    currentBatch++;

                    // Progress Bar
                    var percent = (double)currentBatch / totalBatches;
                    var filled = (int)(percent * 20);
                    var bar = new string('█', filled).PadRight(20, '░');
                    
                    Console.Write($"\r[{bar}] %{percent*100:F1} | Batch: {currentBatch}/{totalBatches} | Loss: {trainer.LastLoss:F4}");
                }

                // Her epoch sonunda checkpoint al
                Console.WriteLine(); 
                CheckpointManager.Save(model, $"checkpoints/model_epoch_{epoch}.ckpt");
                CheckpointManager.Save(model, "checkpoints/model_latest.ckpt");
                Console.WriteLine($"\nCheckpoint: model_epoch_{epoch}.ckpt kaydedildi.");
            }
            model.Mode = ModelMode.Inference;
            model.DisableGradients();
            // model.Quantize();

            CheckpointManager.Save(model, "model_final.ckpt");

            Console.WriteLine($"Training tamamlandı. {totalBatches} batch işlendi.");
        }
    }
}
