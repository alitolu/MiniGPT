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
        Linear embed;
        Linear head;

        TransformerBlock[] blocks;

        public int Dim;
        int vocabSize;

        public int BlockCount => blocks.Length;
        public int Heads = 4; // Default heads, should be passed in ctor but hardcoded for now due to legacy ctor signature

        public MiniGPTModel(int vocab,int d,int layers=2, int heads=4)
        {
            Dim=d;
            vocabSize=vocab;
            Heads=heads;

            embed=new Linear(vocab,d);

            blocks=Enumerable.Range(0,layers)
                .Select(_=>new TransformerBlock(d))
                .ToArray();

            head=new Linear(d,vocab);
        }

        public Tensor Forward(Tensor x, KVCache[] caches=null)
        {
            var h=embed.Forward(x);

            var pe=PositionalEncoding.Build(h.Rows,Dim);
            h=Ops.Add(h,pe);

            for(int i=0;i<blocks.Length;i++)
            {
                var cache=caches==null?null:caches[i];
                h=blocks[i].Forward(h,cache);
            }

            return head.Forward(h);
        }

        public float[] Forward(int[] tokens)
        {
            var x=new Tensor(tokens.Length,vocabSize);
            for(int i=0;i<tokens.Length;i++)
                if(tokens[i]<vocabSize)
                    x[i,tokens[i]]=1.0f;

            var output=Forward(x);

            int lastRow=output.Rows-1;
            float[] logits=new float[vocabSize];
            for(int i=0;i<vocabSize;i++)
                logits[i]=output[lastRow,i];

            return logits;
        }

        Tensor lastOutput;

        public void Backward(int target)
        {
            if(lastOutput==null) return;

            float[] softmax=new float[vocabSize];
            float max=float.MinValue;

            for(int i=0;i<vocabSize;i++)
            {
                float v=lastOutput[lastOutput.Rows-1,i];
                if(v>max) max=v;
            }

            float sum=0;
            for(int i=0;i<vocabSize;i++)
            {
                softmax[i]=MathF.Exp(lastOutput[lastOutput.Rows-1,i]-max);
                sum+=softmax[i];
            }
            for(int i=0;i<vocabSize;i++)
                softmax[i]/=sum;

            softmax[target]-=1.0f;

            if(lastOutput.Grad==null)
                lastOutput.Grad=new float[lastOutput.Data.Length];

            int offset=(lastOutput.Rows-1)*vocabSize;
            for(int i=0;i<vocabSize;i++)
                lastOutput.Grad[offset+i]=softmax[i];
        }

        public List<Tensor> Parameters()
            => embed.Parameters().Concat(head.Parameters()).ToList();

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
                result[i]=(float[])all[i].Data.Clone();
            return result;
        }

        public void ImportWeights(float[][] weights)
        {
            var all=Parameters();
            for(int i=0;i<all.Count && i<weights.Length;i++)
                Array.Copy(weights[i],all[i].Data,all[i].Data.Length);
        }
    }
}
