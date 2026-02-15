using MiniGPT.Core;
using MiniGPT.Model;
using MiniGPT.Optim;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Engine
{
    public class Trainer
    {
        MiniGPTModel model;
        AdamW optim;
        int vocabSize;

        public Trainer(MiniGPTModel m, int vocabSize)
        {
            model=m;
            this.vocabSize=vocabSize;
            optim=new AdamW(model.Parameters(), 0.0003f);
        }

        float CrossEntropy(float[] logits, int target)
        {
            float max = logits.Max();

            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
                sum += MathF.Exp(logits[i] - max);

            float logProb = logits[target] - max - MathF.Log(sum);

            return -logProb;
        }

        public void TrainStep()
        {
            var x=Tensor.Rand(1,vocabSize);
            var y=new Tensor(1,vocabSize);

            var pred=model.Forward(x);

            for(int i=0;i<vocabSize;i++)
                pred.Grad[i]=pred.Data[i]-y.Data[i];

            optim.Step();
            optim.ZeroGrad();
        }

        public void TrainBatch(List<int[]> batch)
        {
            float totalLoss = 0;

            foreach (var tokens in batch)
            {
                for (int i = 0; i < tokens.Length - 1; i++)
                {
                    var input = tokens.Take(i + 1).ToArray();
                    int target = tokens[i + 1];

                    var logits = model.Forward(input);

                    float loss = CrossEntropy(logits, target);
                    totalLoss += loss;

                    model.Backward(target);
                }
            }

            optim.Step();
            optim.ZeroGrad();

            Console.WriteLine($"Loss: {totalLoss / batch.Count:F4}");
        }
    }
}
