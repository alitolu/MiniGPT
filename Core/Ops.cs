using System;
using System.Threading.Tasks;
using System.Numerics;

namespace MiniGPT.Core
{
    public static class Ops
    {
        public static Tensor Add(Tensor a, Tensor b)
        {
            // Matched Shapes (Parallel Add)
            if (a.Rows == b.Rows && a.Cols == b.Cols)
            {
                var c = new Tensor(a.Rows, a.Cols, a.RequiresGrad || b.RequiresGrad);
                System.Threading.Tasks.Parallel.For(0, c.Data.Length, i =>
                {
                    c.Data[i] = a.Data[i] + b.Data[i];
                });
                return c;
            }
            // Broadcasting Bias (MxN + 1xN) - Parallel
            else if (b.Rows == 1 && b.Cols == a.Cols)
            {
                var c = new Tensor(a.Rows, a.Cols, a.RequiresGrad || b.RequiresGrad);
                int cols = a.Cols;
                
                System.Threading.Tasks.Parallel.For(0, a.Rows, i =>
                {
                    int offset = i * cols;
                    // Vector Add per row
                    for (int j = 0; j < cols; j++)
                    {
                        c.Data[offset + j] = a.Data[offset + j] + b.Data[j];
                    }
                });
                return c;
            }
            else
            {
                throw new Exception($"Shape mismatch in Add: {a.Rows}x{a.Cols} vs {b.Rows}x{b.Cols}");
            }
        }

        public static Tensor Sub(Tensor a, Tensor b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                 throw new Exception($"Shape mismatch in Sub: {a.Rows}x{a.Cols} vs {b.Rows}x{b.Cols}");

            var o = new Tensor(a.Rows, a.Cols, true);
            for(int i=0; i<o.Data.Length; i++)
                o.Data[i] = a.Data[i] - b.Data[i];
            return o;
        }

        public static Tensor Mul(Tensor a, Tensor b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                 throw new Exception($"Shape mismatch in Mul: {a.Rows}x{a.Cols} vs {b.Rows}x{b.Cols}");

            var o = new Tensor(a.Rows, a.Cols, true);
            for(int i=0; i<o.Data.Length; i++)
                o.Data[i] = a.Data[i] * b.Data[i];
            return o;
        }

        public static Tensor Div(Tensor a, Tensor b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                 throw new Exception($"Shape mismatch in Div: {a.Rows}x{a.Cols} vs {b.Rows}x{b.Cols}");

            var o = new Tensor(a.Rows, a.Cols, true);
            for(int i=0; i<o.Data.Length; i++)
                o.Data[i] = a.Data[i] / (b.Data[i] + 1e-8f); // Epsilon
            return o;
        }

        public static Tensor MatMul(Tensor a, Tensor b)
        {
            if(a.Cols!=b.Rows) throw new Exception($"MatMul Shape Error: {a.Rows}x{a.Cols} * {b.Rows}x{b.Cols}");
            
            var c=new Tensor(a.Rows,b.Cols,true);

            // Transpose b for cache locality (row-major access)
            // This makes k loop sequential for bT as well as a
            var bT = Transpose(b);

            Parallel.For(0, a.Rows, i =>
            {
                // Pre-calculate offset for row i of a
                int offsetA = i * a.Cols;

                for(int j=0;j<b.Cols;j++)
                {
                    // Pre-calculate offset for row j of bT (column j of b)
                    int offsetB = j * bT.Cols; // bT.Cols == b.Rows == a.Cols

                    float sum=0;
                    for(int k=0;k<a.Cols;k++)
                    {
                        // Direct array access is faster than indexer if possible, but let's trust JIT for now.
                        // Ideally: sum += a.Data[offsetA + k] * bT.Data[offsetB + k];
                        // Using public indexer:
                        // sum += a[i,k] * bT[j,k];
                        
                        // Let's use direct data access for speed if Data is public
                        sum += a.Data[offsetA + k] * bT.Data[offsetB + k];
                    }
                    c.Data[i * c.Cols + j] = sum;
                }
            });

            return c;
        }

        public static Tensor Transpose(Tensor a)
        {
            var o = new Tensor(a.Cols, a.Rows, true);
            for(int i=0; i<a.Rows; i++)
                for(int j=0; j<a.Cols; j++)
                    o[j,i] = a[i,j];
            return o;
        }

        public static Tensor ReLU(Tensor a)
        {
            var o=new Tensor(a.Rows,a.Cols,true);
            for(int i=0;i<o.Data.Length;i++)
                o.Data[i]=Math.Max(0, a.Data[i]);
            return o;
        }

        public static Tensor Softmax(Tensor x)
        {
            var o=new Tensor(x.Rows,x.Cols,true);

            for(int i=0;i<x.Rows;i++)
            {
                float max=float.MinValue;
                for(int j=0;j<x.Cols;j++) 
                    if(x[i,j]>max) max=x[i,j];

                float sum=0;
                for(int j=0;j<x.Cols;j++)
                {
                    float e=(float)Math.Exp(x[i,j]-max);
                    o[i,j]=e;
                    sum+=e;
                }

                for(int j=0;j<x.Cols;j++) o[i,j]/=sum;
            }
            return o;
        }
        
        // Scalar Ops
        public static Tensor Mul(Tensor a, float s)
        {
             var o=new Tensor(a.Rows,a.Cols,true);
             for(int i=0;i<o.Data.Length;i++) o.Data[i]=a.Data[i]*s;
             return o;
        }
        
        public static Tensor Add(Tensor a, float s)
        {
             var o=new Tensor(a.Rows,a.Cols,true);
             for(int i=0;i<o.Data.Length;i++) o.Data[i]=a.Data[i]+s;
             return o;
        }
        
        public static Tensor Sqrt(Tensor a)
        {
             var o=new Tensor(a.Rows,a.Cols,true);
             for(int i=0;i<o.Data.Length;i++) o.Data[i]=(float)Math.Sqrt(a.Data[i]);
             return o;
        }

        public static Tensor Tanh(Tensor a)
        {
             var o=new Tensor(a.Rows,a.Cols,true);
             for(int i=0;i<o.Data.Length;i++) o.Data[i]=(float)Math.Tanh(a.Data[i]);
             return o;
        }

        // Random & Init
        public static Tensor Random(int rows, int cols, float scale=1.0f)
        {
            var t = new Tensor(rows, cols);
            var rnd = new Random();
            for(int i=0; i<t.Data.Length; i++)
                t.Data[i] = ((float)rnd.NextDouble() * 2 - 1) * scale;
            return t;
        }

        public static Tensor Zeros(int rows, int cols)
        {
            return new Tensor(rows, cols, true); // true=zeros
        }
        public static Tensor SumRows(Tensor a)
        {
            // (Rows, Cols) -> (1, Cols) sum over rows
            var o = new Tensor(1, a.Cols, true);
            for(int i=0; i<a.Rows; i++)
                for(int j=0; j<a.Cols; j++)
                    o.Data[j] += a.Data[i * a.Cols + j];
            return o;
        }
        public static Tensor ReLUBackward(Tensor gradOutput, Tensor input)
        {
            var dx = new Tensor(gradOutput.Rows, gradOutput.Cols, true);
            for(int i=0; i<gradOutput.Data.Length; i++)
            {
                dx.Data[i] = input.Data[i] > 0 ? gradOutput.Data[i] : 0;
            }
            return dx;
        }
    } 
}
