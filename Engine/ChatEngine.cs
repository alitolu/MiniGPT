using MiniGPT.Tokenizer;
using MiniGPT.Model;
using MiniGPT.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
            var ids=tok.Encode(input); // List<int>
            var o=model.Forward(ids.ToArray());

            int best=0;
            // o is float[] here because Forward(int[]) calls the wrapper
            float max=o[0];

            for(int i=1;i<o.Length;i++)
                if(o[i]>max){max=o[i];best=i;}

            return tok.Decode(new[]{best});
        }

        public string Generate(string prompt,int maxTokens=50)
        {
            var tokens=tok.Encode(prompt); // List<int>

            var caches=new KVCache[model.BlockCount];
            // Dim per head = ModelDim / Heads
            int headDim = model.Dim / model.Heads; 
            
            for(int i=0;i<caches.Length;i++)
                caches[i]=new KVCache(headDim);

            // Prefill: Feed prompt into model to populate cache
            if(tokens.Count > 0)
            {
               var logits = model.Forward(tokens.ToArray(), caches);
               
               // Predict next token from last logits
               var lastRow = logits.Rows - 1;
               float[] nextLogits = new float[vocab];
               for(int j=0; j<vocab; j++) nextLogits[j] = logits[lastRow, j];
               
               int next = Sample(nextLogits, 0.8f);
               if(next == tok.EOS) return tok.Decode(tokens);
               
               tokens.Add(next);
            }

            // Generation Loop
            for(int step=0; step < maxTokens-1; step++)
            {
                var lastToken=new int[]{tokens[^1]};

                var logits=model.Forward(lastToken, caches);

                float[] last=new float[vocab];
                for(int i=0;i<vocab;i++)
                    last[i]=logits[0,i];

                int next=Sample(last,0.8f);

                if(next==tok.EOS)
                    break;

                tokens.Add(next);
            }

            return tok.Decode(tokens);
        }

        int Sample(float[] logits,float temperature)
        {
            for(int i=0;i<logits.Length;i++)
                logits[i]/=temperature;

            float max=logits.Max();

            Parallel.For(0,logits.Length,i=>
            {
                logits[i]=(float)Math.Exp(logits[i]-max);
            });

            float sum=logits.Sum();
            for(int i=0;i<logits.Length;i++)
                logits[i]/=sum;

            float r=(float)Random.Shared.NextDouble();
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
