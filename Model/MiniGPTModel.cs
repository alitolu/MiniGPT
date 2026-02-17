using MiniGPT.Core;
using MiniGPT.NN;
using MiniGPT.Engine;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Model
{
    public class MiniGPTModel
    {
        Embedding embed;
        Linear head;

        TransformerBlock[] blocks;

        public int Dim;
        int vocabSize;

        public int BlockCount => blocks.Length;
        public int Heads = 4; 

        Tensor lastOutput; // For Backward

        public MiniGPTModel(int vocab,int d,int layers=2, int heads=4)
        {
            Dim=d;
            vocabSize=vocab;
            Heads=heads;
            
            embed=new Embedding(vocab,d);

            blocks=Enumerable.Range(0,layers)
                .Select(_=>new TransformerBlock(d))
                .ToArray();

            head=new Linear(d,vocab);
        }

        public Tensor Forward(int[] tokens, KVCache[] caches=null)
        {
            // Optimized Embedding Lookup
            var h=embed.Forward(tokens);

            var pe=PositionalEncoding.Build(h.Rows,Dim);
            h=Ops.Add(h,pe);

            for(int i=0;i<blocks.Length;i++)
            {
                var cache=caches==null?null:caches[i];
                h=blocks[i].Forward(h,cache);
            }
            
            var outTensor = head.Forward(h);
            
            if (caches == null) lastOutput = outTensor;
            
            return outTensor;
        }

        public float[] Forward(int[] tokens)
        {
            // Wrapper for simple inference (returns logits of last token)
            var output = Forward(tokens, null);

            int lastRow=output.Rows-1;
            float[] logits=new float[vocabSize];
            for(int i=0;i<vocabSize;i++)
                logits[i]=output[lastRow,i];

            return logits;
        }

        public void Backward(Tensor gradOutput)
        {
            // Full Sequence Backward Pass
            // Chain Rule: Head -> Blocks -> Embedding

            // 1. Head Backward
            var dH = head.Backward(gradOutput);

            // 2. Transformer Blocks Backward (Reverse Order)
            for(int i=blocks.Length-1; i>=0; i--)
            {
                dH = blocks[i].Backward(dH);
            }

            // 3. Positional Encoding Backward (Pass-through for Addition)
            // dLoss/d(h+pe) = dLoss/dh * 1 + dLoss/dpe * 1
            // Since PE is fixed, we only propagate to Embedding
            
            // 4. Embedding Backward
            // Note: Embedding is implemented as Linear(OneHot), so standard Linear backward works.
            embed.Backward(dH);
        }
        
        // This Backward sets the gradient of the output.
        // The Trainer usually calls backwards on the graph.
        // Assuming Trainer handles tensor backward propagation if specificied.
        
        public List<Tensor> Parameters()
        {
             var p = new List<Tensor>();
             p.AddRange(embed.Parameters());
             foreach(var b in blocks) p.AddRange(b.Parameters());
             p.AddRange(head.Parameters());
             return p;
        }

        public void Quantize()
        {
            embed.Quantize();
            head.Quantize();

            foreach(var b in blocks)
                b.Quantize();
        }

        public ModelMode Mode = ModelMode.Train;

        public void DisableGradients()
        {
            foreach(var p in Parameters())
                p.RequiresGrad=false;
        }

        public float[][] ExportWeights()
        {
            var all=Parameters();
            var result=new float[all.Count][];
            for(int i=0;i<all.Count;i++)
            {
                if(all[i] == null || all[i].Data == null) 
                   continue; // Safety check
                   
                result[i]=(float[])all[i].Data.Clone();
            }
            return result;
        }

        public void ImportWeights(float[][] weights)
        {
            var all=Parameters();
            for(int i=0;i<all.Count && i<weights.Length;i++)
            {
                if(all[i] == null || all[i].Data == null) continue;
                if(weights[i] == null) continue;
                
                if (weights[i].Length == all[i].Data.Length)
                    Array.Copy(weights[i],all[i].Data,all[i].Data.Length);
            }
        }
        
        public string Generate(string prompt) { return ""; } // Placeholder for Agent Loop compilation
    }
}
