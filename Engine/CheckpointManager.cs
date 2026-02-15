using MiniGPT.Core;
using MiniGPT.Model;
using System.IO;
using System.Text.Json;

namespace MiniGPT.Engine
{
    public static class CheckpointManager
    {
        public static void Save(MiniGPTModel model, string path)
        {
            var data = model.ExportWeights();
            var json = JsonSerializer.Serialize(data);
            File.WriteAllText(path, json);
        }

        public static void Load(MiniGPTModel model, string path)
        {
            var json = File.ReadAllText(path);
            var weights = JsonSerializer.Deserialize<float[][]>(json);
            model.ImportWeights(weights);
        }
    }
}
