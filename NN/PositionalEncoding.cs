using MiniGPT.Core;
using System;

namespace MiniGPT.NN
{
    public static class PositionalEncoding
    {
        public static Tensor Build(int seq,int dim)
        {
            var pe=new Tensor(seq,dim);

            for(int pos=0;pos<seq;pos++)
            for(int i=0;i<dim;i+=2)
            {
                float div=(float)Math.Pow(10000,i/(float)dim);

                pe[pos,i]=(float)Math.Sin(pos/div);

                if(i+1<dim)
                    pe[pos,i+1]=(float)Math.Cos(pos/div);
            }

            return pe;
        }
    }
}
