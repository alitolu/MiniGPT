using MiniGPT.Core;
using System.Collections.Generic;

namespace MiniGPT.NN
{
    public class Linear
    {
        public Tensor W;
        public Tensor B;
        QuantizedTensor qW;

        public Linear(int input,int output)
        {
            W=Tensor.Rand(input,output,true);
            B=Tensor.Rand(1,output,true);
        }

        public void Quantize()
        {
            qW=new QuantizedTensor(W);
        }

        public Tensor Forward(Tensor x)
        {
            int rows=x.Rows;
            int cols=W.Cols;
            int inner=x.Cols;

            var mm=new Tensor(rows,cols,true);

            for(int i=0;i<rows;i++)
                for(int j=0;j<cols;j++)
                    for(int k=0;k<inner;k++)
                    {
                        float w=qW!=null
                            ? qW.Get(k*cols+j)
                            : W[k,j];
                        mm[i,j]+=x[i,k]*w;
                    }

            return Ops.Add(mm,B);
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return W;
            yield return B;
        }
    }
}
