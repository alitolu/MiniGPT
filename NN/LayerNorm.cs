using MiniGPT.Core;
using System;
using System.Collections.Generic;

namespace MiniGPT.NN
{
    public class LayerNorm
    {
        int dim;
        float eps = 1e-5f;

        public Tensor Gamma; // Scale
        public Tensor Beta;  // Shift

        public LayerNorm(int d)
        {
            dim = d;
            // Gamma init 1, Beta init 0
            Gamma = new Tensor(1, d, true);
            for(int i=0; i<d; i++) Gamma.Data[i] = 1.0f;
            
            Beta = new Tensor(1, d, true);
            // Beta default 0
        }

        Tensor lastInput;

        public Tensor Forward(Tensor x)
        {
            lastInput = x;
            var o = new Tensor(x.Rows, x.Cols, true);
            
            for(int r=0; r<x.Rows; r++)
            {
                float mean = 0;
                for(int c=0; c<dim; c++) mean += x.Data[r*dim+c];
                mean /= dim;

                float var = 0;
                for(int c=0; c<dim; c++) {
                     float diff = x.Data[r*dim+c] - mean;
                     var += diff*diff;
                }
                var /= dim;
                
                float inv = 1.0f / (float)Math.Sqrt(var + eps);
                
                int offset = r * dim;
                for(int c=0; c<dim; c++)
                {
                    float norm = (x.Data[offset+c] - mean) * inv;
                    o.Data[offset+c] = norm * Gamma.Data[c] + Beta.Data[c];
                }
            }

            return o;
        }

        public Tensor Backward(Tensor gradOutput)
        {
            int rows = gradOutput.Rows;
            int cols = gradOutput.Cols;
            var dx = new Tensor(rows, cols, true);

            if(Gamma.Grad==null) Gamma.Grad=new float[dim];
            if(Beta.Grad==null) Beta.Grad=new float[dim];

            for(int r=0; r<rows; r++)
            {
                int offset = r * cols;
                
                // Recompute Mean/InvStd/Norm for this row
                float mean = 0;
                for(int c=0; c<dim; c++) mean += lastInput.Data[offset+c];
                mean /= dim;
                
                float var = 0;
                for(int c=0; c<dim; c++) {
                    float diff = lastInput.Data[offset+c] - mean;
                    var += diff*diff;
                }
                var /= dim;
                float invStd = 1.0f / (float)Math.Sqrt(var + eps);

                float sumGradGamma = 0;
                float sumGradBeta = 0;
                float sumGradX = 0;
                float sumGradXNorm = 0;

                for(int c=0; c<dim; c++)
                {
                    float g = gradOutput.Data[offset+c];
                    float x_hat = (lastInput.Data[offset+c] - mean) * invStd;
                    
                    // Gradients w.r.t params
                    Gamma.Grad[c] += g * x_hat;
                    Beta.Grad[c] += g;

                    // Intermediate for dx
                    float dx_hat = g * Gamma.Data[c];
                    sumGradX += dx_hat;
                    sumGradXNorm += dx_hat * x_hat;
                }

                for(int c=0; c<dim; c++)
                {
                    float x_hat = (lastInput.Data[offset+c] - mean) * invStd;
                    float dx_hat = gradOutput.Data[offset+c] * Gamma.Data[c];
                    
                    // Standard LayerNorm Backward Formula
                    float term1 = dx_hat;
                    float term2 = sumGradX / dim;
                    float term3 = x_hat * (sumGradXNorm / dim);
                    
                    dx.Data[offset+c] = invStd * (term1 - term2 - term3);
                }
            }
            return dx;
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }
    }
}
