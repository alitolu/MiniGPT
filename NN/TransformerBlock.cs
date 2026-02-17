using MiniGPT.Core;
using MiniGPT.Engine;
using System.Collections.Generic;

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

        Tensor lastFc1Out; // For ReLU backward

        public Tensor Forward(Tensor x, KVCache cache=null)
        {
            // Note: We need to save intermediates for backward if training
            // Since layers save their own inputs, we mostly need to manage flow
             
            var h1=attn.Forward(ln1.Forward(x),cache);
            var res1=Ops.Add(x,h1); 

            var ln2Out = ln2.Forward(res1);
            var fc1Out = fc1.Forward(ln2Out);
            lastFc1Out = fc1Out; // Save for ReLU backward

            var h2=fc2.Forward(Ops.ReLU(fc1Out));

            var res2=Ops.Add(res1,h2);
            return res2;
        }

        public Tensor Backward(Tensor gradOutput)
        {
            // 1. Skip Connection 2 (Add) -> gradOutput goes to both path
            // Path B: MLP
            var dReluOut = fc2.Backward(gradOutput);
            var dFc1Out = Ops.ReLUBackward(dReluOut, lastFc1Out);
            var dLn2Out = fc1.Backward(dFc1Out);
            var dRes1_FromMLP = ln2.Backward(dLn2Out);
            
            // Path A: Skip -> gradOutput
            // Sum gradients at Res1 node
            var dRes1 = Ops.Add(gradOutput, dRes1_FromMLP);

            // 2. Skip Connection 1 (Add) -> dRes1 goes to both path
            // Path B: Attention
            var dLn1Out = attn.Backward(dRes1);
            var dX_FromAttn = ln1.Backward(dLn1Out);

            // Path A: Skip -> dRes1
            // Sum gradients at Input node
            var dX = Ops.Add(dRes1, dX_FromAttn);

            return dX;
        }

        public IEnumerable<Tensor> Parameters()
        {
             foreach(var p in ln1.Parameters()) yield return p;
             foreach(var p in attn.Parameters()) yield return p;
             foreach(var p in ln2.Parameters()) yield return p;
             foreach(var p in fc1.Parameters()) yield return p;
             foreach(var p in fc2.Parameters()) yield return p;
        }

        public void Quantize()
        {
            attn.Quantize();
            fc1.Quantize();
            fc2.Quantize();
        }
    }
}
