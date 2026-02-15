using MiniGPT.Model;
using System.IO;

namespace MiniGPT.Export
{
    public static class GGUFExporter
    {
        public static void Export(
            MiniGPTModel model,
            string path)
        {
            using var bw = new BinaryWriter(File.Create(path));

            bw.Write(System.Text.Encoding.ASCII.GetBytes("GGUF"));
            bw.Write(1); // version

            var weights = model.ExportWeights();

            bw.Write(weights.Length);

            foreach(var w in weights)
            {
                bw.Write(w.Length);
                foreach(var val in w)
                    bw.Write(val);
            }
        }
    }
}
