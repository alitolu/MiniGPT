using MiniGPT.Core;
using System;

namespace MiniGPT.NN
{
    public class LayerNorm
    {
        int dim;
        float eps = 1e-5f;

        public LayerNorm(int d)
        {
            dim = d;
        }

        public Tensor Forward(Tensor x)
        {
            var o = new Tensor(x.Rows, x.Cols, true);

            for(int r=0;r<x.Rows;r++)
            {
                float mean=0;
                for(int c=0;c<dim;c++)
                    mean+=x[r,c];
                mean/=dim;

                float var=0;
                for(int c=0;c<dim;c++)
                    var+=(x[r,c]-mean)*(x[r,c]-mean);
                var/=dim;

                float inv=(float)(1.0/Math.Sqrt(var+eps));

                for(int c=0;c<dim;c++)
                    o[r,c]=(x[r,c]-mean)*inv;
            }

            return o;
        }
    }
}
