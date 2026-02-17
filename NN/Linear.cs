using MiniGPT.Core;
using System.Collections.Generic;

namespace MiniGPT.NN
{
    public class Linear
    {
        public Tensor W;
        public Tensor B;
        Q4Tensor qW;

        public Linear(int input,int output)
        {
            W=Tensor.Rand(input,output,true);
            B=Tensor.Rand(1,output,true);
        }

        public void Quantize()
        {
            qW=new Q4Tensor(W);
            W=null; // Free float weights
        }

        Tensor lastInput;
        public Tensor Forward(Tensor x)
        {
            lastInput = x;
            int rows=x.Rows;
            int cols=qW!=null ? qW.Cols : W.Cols;
            int inner=x.Cols;

            var mm=new Tensor(rows,cols,true);

            // MatMul Optimization (Transpose Trick)
            var wT = Ops.Transpose(W); // Transpose Weights for Cache Locality

            System.Threading.Tasks.Parallel.For(0, rows, i =>
            {
                int offsetA = i * inner;
                for(int j=0; j<cols; j++)
                {
                    float sum=0;
                    int offsetB = j * inner; // wT row j = W col j
                    for(int k=0; k<inner; k++)
                    {
                        sum += x.Data[offsetA + k] * wT.Data[offsetB + k];
                    }
                    mm.Data[i*cols + j] = sum;
                }
            });

            return Ops.Add(mm,B);
        }

        public Tensor Backward(Tensor gradOutput)
        {
            // dL/dW = x^T * gradOutput
            // dL/db = sum(gradOutput, axis=0)
            // dL/dx = gradOutput * W^T

            if(W.Grad==null) W.Grad=new float[W.Data.Length];
            if(B.Grad==null) B.Grad=new float[B.Data.Length];

            var xT = Ops.Transpose(lastInput);
            var dW = Ops.MatMul(xT, gradOutput);

            // Accumulate Gradients
            for(int i=0; i<dW.Data.Length; i++) 
                W.Grad[i] += dW.Data[i];

            var dB = Ops.SumRows(gradOutput);
            for(int i=0; i<dB.Data.Length; i++) 
                B.Grad[i] += dB.Data[i];

            var wT = Ops.Transpose(W);
            var dX = Ops.MatMul(gradOutput, wT);

            return dX;
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return W;
            yield return B;
        }
    }
}
