using System;

namespace MiniGPT.Core
{
    public static class Ops
    {
        public static Tensor Add(Tensor a, Tensor b)
        {
            var o=new Tensor(a.Rows,a.Cols,true);

            for(int i=0;i<o.Data.Length;i++)
                o.Data[i]=a.Data[i]+b.Data[i];

            return o;
        }

        public static Tensor MatMul(Tensor a, Tensor b)
        {
            var o=new Tensor(a.Rows,b.Cols,true);

            for(int i=0;i<a.Rows;i++)
                for(int j=0;j<b.Cols;j++)
                    for(int k=0;k<a.Cols;k++)
                        o[i,j]+=a[i,k]*b[k,j];

            return o;
        }

        public static Tensor ReLU(Tensor x)
        {
            var o=new Tensor(x.Rows,x.Cols,true);

            for(int i=0;i<x.Data.Length;i++)
                o.Data[i]=Math.Max(0,x.Data[i]);

            return o;
        }
    }
}
