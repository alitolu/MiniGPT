using MiniGPT.Core;
using MiniGPT.NN;
using MiniGPT.Engine;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Model
{
    public class MiniGPTModel
    {
        Linear embed;
        Linear head;

        TransformerBlock[] blocks;

        int dim;

        public int BlockCount => blocks.Length;

        public MiniGPTModel(int vocab,int d,int layers=2)
        {
            dim=d;

            embed=new Linear(vocab,d);

            blocks=Enumerable.Range(0,layers)
                .Select(_=>new TransformerBlock(d))
                .ToArray();

            head=new Linear(d,vocab);
        }

        public Tensor Forward(Tensor x, KVCache[] caches=null)
        {
            var h=embed.Forward(x);

            var pe=PositionalEncoding.Build(h.Rows,dim);
            h=Ops.Add(h,pe);

            for(int i=0;i<blocks.Length;i++)
            {
                var cache=caches==null?null:caches[i];
                h=blocks[i].Forward(h,cache);
            }

            return head.Forward(h);
        }

        public IEnumerable<Tensor> Parameters()
            => embed.Parameters().Concat(head.Parameters());
    }
}
