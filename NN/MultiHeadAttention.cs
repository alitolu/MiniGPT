using MiniGPT.Core;
using MiniGPT.Engine;
using System.Collections.Generic;

namespace MiniGPT.NN
{
    public class MultiHeadAttention
    {
        Linear Wq,Wk,Wv,Wo;
        int dim;
        bool useFlash=true;

        public MultiHeadAttention(int d)
        {
            dim=d;
            Wq=new Linear(d,d);
            Wk=new Linear(d,d);
            Wv=new Linear(d,d);
            Wo=new Linear(d,d);
        }

        Tensor lastInput, lastQ, lastK, lastV, lastScores;

        public Tensor Forward(Tensor x, KVCache cache=null)
        {
            lastInput = x;
            var Q=Wq.Forward(x);
            var K=Wk.Forward(x);
            var V=Wv.Forward(x);

            lastQ = Q; lastK = K; lastV = V;

            if(cache!=null)
            {
                // Inference Mode (Generation) - No Backward needed
                cache.Add(K,V); 
                return Wo.Forward(FlashAttention.Compute(Q,cache));
            }
            else
            {
                // Training Mode - Use Classic Attention for Backward Support
                var context = ClassicAttention(Q,K,V);
                return Wo.Forward(context);
            }
        }

        Tensor ClassicAttention(Tensor Q,Tensor K,Tensor V)
        {
            var scores=new Tensor(Q.Rows,K.Rows,true);

            // 1. Scores = Q * K^T / sqrt(d)
            float scale = 1.0f / (float)System.Math.Sqrt(dim);
            
            // Masking & Softmax
            for(int i=0;i<Q.Rows;i++)
            {
                float max = float.MinValue;
                for(int j=0;j<K.Rows;j++)
                {
                    if(j > i) { scores[i,j] = float.NegativeInfinity; continue; } // Causal Mask

                    float s=0;
                    for(int k=0;k<dim;k++)s+=Q[i,k]*K[j,k];
                    s *= scale;
                    scores[i,j]=s;
                    if(s>max) max=s;
                }
                
                // Softmax
                float sum=0;
                for(int j=0;j<K.Rows;j++)
                {
                    if(j > i) { scores[i,j]=0; continue; }
                    
                    float e=(float)System.Math.Exp(scores[i,j]-max);
                    scores[i,j]=e;
                    sum+=e;
                }
                for(int j=0;j<K.Rows;j++) scores[i,j]/=sum;
            }
            
            lastScores = scores; // Save for backward

            // 2. Output = Scores * V
            var outv=new Tensor(Q.Rows,dim,true);
            for(int i=0;i<Q.Rows;i++)
            for(int j=0;j<K.Rows;j++)
            {
                 // Optimization: Skip zero scores
                 float s = scores[i,j];
                 if(s == 0) continue;
                 
                 for(int k=0;k<dim;k++)
                    outv[i,k]+=s*V[j,k];
            }

            return outv;
        }

        public Tensor Backward(Tensor gradOutput)
        {
            // 1. Backward through Wo
            var dContext = Wo.Backward(gradOutput);

            int n = dContext.Rows;
            int d = dim;
            float scale = 1.0f / (float)System.Math.Sqrt(d);

            var dQ = new Tensor(n, d, true);
            var dK = new Tensor(n, d, true);
            var dV = new Tensor(n, d, true);
            var dScores = new Tensor(n, n, true);

            // 2. Backward through Context = Scores * V
            // dV = Scores^T * dContext
            // dScores = dContext * V^T
            
            for(int i=0; i<n; i++)
            for(int j=0; j<n; j++)
            {
                float score = lastScores[i,j];
                
                // dV accumulator
                for(int k=0; k<d; k++) dV[j,k] += score * dContext[i,k];
                
                // dScores accumulator
                float dS = 0;
                for(int k=0; k<d; k++) dS += dContext[i,k] * lastV[j,k];
                dScores[i,j] = dS;
            }

            // 3. Backward through Softmax (Jacobian)
            // dS_pre_softmax = Score * (dScore - sum(dScore * Score))
            for(int i=0; i<n; i++)
            {
                float sumDS = 0;
                for(int j=0; j<n; j++) sumDS += dScores[i,j] * lastScores[i,j];

                for(int j=0; j<n; j++)
                {
                    if(j>i) continue; // Masked
                    float s = lastScores[i,j];
                    float dS = s * (dScores[i,j] - sumDS);
                    dScores[i,j] = dS * scale; // Include scaling factor here
                }
            }

            // 4. Backward through Q * K^T
            // dQ = dScores * K
            // dK = dScores^T * Q
            
            for(int i=0; i<n; i++)
            for(int j=0; j<n; j++)
            {
                if(j > i) continue; // Skip masked
                
                float dS = dScores[i,j];
                
                for(int k=0; k<d; k++)
                {
                    dQ[i,k] += dS * lastK[j,k];
                    dK[j,k] += dS * lastQ[i,k];
                }
            }

            // 5. Backward through Projections (Wq, Wk, Wv)
            var dxQ = Wq.Backward(dQ);
            var dxK = Wk.Backward(dK);
            var dxV = Wv.Backward(dV);
            
            // 6. Sum gradients for Input
            var dx = Ops.Add(dxQ, Ops.Add(dxK, dxV));
            return dx;
        }

        public IEnumerable<Tensor> Parameters()
        {
             foreach(var p in Wq.Parameters()) yield return p;
             foreach(var p in Wk.Parameters()) yield return p;
             foreach(var p in Wv.Parameters()) yield return p;
             foreach(var p in Wo.Parameters()) yield return p;
        }

        public void Quantize()
        {
            Wq.Quantize();
            Wk.Quantize();
            Wv.Quantize();
            Wo.Quantize();
        }
    }
}
