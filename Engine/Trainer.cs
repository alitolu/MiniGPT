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

        // Removed legacy TrainStep method that caused compile errors

        public float LastLoss { get; private set; }

        public void TrainBatch(List<int[]> batch)
        {
          try {
            float totalLoss = 0;
            int totalTokens = 0;

            int batchIdx = 0;
            foreach (var tokens in batch)
            {
                batchIdx++;
                // Console.WriteLine($"Processing Sequence {batchIdx}/{batch.Count} Len:{tokens.Length}");
                
                if(tokens.Length < 2) continue;
                
                int seqLen = tokens.Length - 1;
                int[] inputTokens = new int[seqLen];
                Array.Copy(tokens, 0, inputTokens, 0, seqLen);

                // Console.WriteLine("Forward...");
                var logits = model.Forward(inputTokens, null); 
                // Console.WriteLine("Forward Done.");
                
                var grad = new Tensor(logits.Rows, logits.Cols, true); 
                float seqLoss = 0;

                for(int t=0; t<seqLen; t++)
                {
                    int target = tokens[t+1];
                    if(target >= vocabSize) continue;

                    float max = float.MinValue;
                    int offset = t * vocabSize;
                    
                    for(int j=0; j<vocabSize; j++) 
                    {
                        float val = logits.Data[offset + j];
                        if(val > max) max = val;
                    }

                    float sum = 0;
                    for(int j=0; j<vocabSize; j++) 
                    {
                        float exp = MathF.Exp(logits.Data[offset + j] - max);
                        grad.Data[offset + j] = exp; 
                        sum += exp;
                    }
                    
                    float invSum = 1.0f / sum;
                    for(int j=0; j<vocabSize; j++) 
                    {
                        float prob = grad.Data[offset + j] * invSum;
                        grad.Data[offset + j] = prob; 
                    }

                    float p_target = grad.Data[offset + target];
                    seqLoss -= MathF.Log(p_target + 1e-9f);
                    grad.Data[offset + target] -= 1.0f;
                }

                // Loss Calc...
                
                totalLoss += seqLoss;
                totalTokens += seqLen;

                model.Backward(grad);
            }

            optim.Step();
            optim.ZeroGrad();

            if (totalTokens > 0)
                LastLoss = totalLoss / totalTokens;
          } catch(Exception ex) {
              Console.WriteLine($"\nCRITICAL TRAINING ERROR: {ex.Message}");
              Console.WriteLine(ex.StackTrace);
              throw; 
          }
        }
    }
}
