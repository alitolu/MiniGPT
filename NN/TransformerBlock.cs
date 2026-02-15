using MiniGPT.Core;
using MiniGPT.Engine;

namespace MiniGPT.NN
{
    public class TransformerBlock
    {
        LayerNorm ln1;
        LayerNorm ln2;
        MultiHeadAttention attn;
        Linear fc1,fc2;

        public TransformerBlock(int dim)
        {
            ln1=new LayerNorm(dim);
            ln2=new LayerNorm(dim);

            attn=new MultiHeadAttention(dim);

            fc1=new Linear(dim,dim*4);
            fc2=new Linear(dim*4,dim);
        }

        public Tensor Forward(Tensor x, KVCache cache=null)
        {
            var h1=attn.Forward(ln1.Forward(x),cache);
            x=Ops.Add(x,h1);

            var h2=fc2.Forward(
                Ops.ReLU(fc1.Forward(ln2.Forward(x)))
            );

            x=Ops.Add(x,h2);

            return x;
        }
    }
}
