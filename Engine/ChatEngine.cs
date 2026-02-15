using MiniGPT.Tokenizer;
using MiniGPT.Model;
using MiniGPT.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Engine
{
    public class ChatEngine
    {
        BPETokenizer tok;
        MiniGPTModel model;
        int vocab;

        public ChatEngine(BPETokenizer t,MiniGPTModel m,int vocabSize)
        {
            tok=t;
            model=m;
            vocab=vocabSize;
        }

        public string Reply(string input)
        {
            var ids=tok.Encode(input);

            var x=new Tensor(1,ids.Length);
            for(int i=0;i<ids.Length;i++)
                x[0,i]=ids[i];

            var o=model.Forward(x);

            int best=0;
            float max=o.Data[0];

            for(int i=1;i<o.Data.Length;i++)
                if(o.Data[i]>max){max=o.Data[i];best=i;}

            return tok.Decode(new[]{best});
        }

        public string Generate(string prompt,int maxTokens=30)
        {
            var tokens=tok.Encode(prompt).ToList();

            var caches=new KVCache[model.BlockCount];

            for(int i=0;i<caches.Length;i++)
                caches[i]=new KVCache();

            for(int step=0;step<maxTokens;step++)
            {
                var lastToken=new List<int>{tokens[^1]};

                var x=OneHot(lastToken);

                var logits=model.Forward(x,caches);

                float[] last=new float[vocab];

                for(int i=0;i<vocab;i++)
                    last[i]=logits[0,i];

                int next=Sample(last,0.8f);

                tokens.Add(next);
            }

            return tok.Decode(tokens);
        }

        Tensor OneHot(List<int> ids)
        {
            var t=new Tensor(ids.Count,vocab);
            for(int i=0;i<ids.Count;i++)
                if(ids[i]<vocab)
                    t[i,ids[i]]=1.0f;
            return t;
        }

        int Sample(float[] logits,float temperature)
        {
            var rnd=new Random();

            for(int i=0;i<logits.Length;i++)
                logits[i]/=temperature;

            float max=logits.Max();
            for(int i=0;i<logits.Length;i++)
                logits[i]=(float)Math.Exp(logits[i]-max);

            float sum=logits.Sum();
            for(int i=0;i<logits.Length;i++)
                logits[i]/=sum;

            float r=(float)rnd.NextDouble();
            float cum=0;

            for(int i=0;i<logits.Length;i++)
            {
                cum+=logits[i];
                if(r<cum) return i;
            }

            return logits.Length-1;
        }
    }
}
