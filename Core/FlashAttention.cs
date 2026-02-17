using MiniGPT.Engine;
using System;

namespace MiniGPT.Core
{
    public static class FlashAttention
    {
        // Standart Compute (Training / Prompt Phase)
        public static Tensor Compute(Tensor Q, Tensor K, Tensor V)
        {
            int n = Q.Rows;
            int d = Q.Cols;
            var output = new Tensor(n, d, true);

            for(int i=0; i<n; i++)
            {
                float maxScore = float.MinValue;
                float denom = 0;
                
                // 1. Pass: Find Max
                // (Optimized: combined loops usually better but keep simple for now)
                
                // We can combine max and sum in one pass with "online softmax" 
                // but let's stick to stable 2-pass or safe online.
                // Standard FlashAttention uses tiling. Here we use row-wise streaming.

                // Calculate scores
                for(int j=0; j<K.Rows; j++)
                {
                    float score = 0;
                    for(int k=0; k<d; k++) score += Q[i,k] * K[j,k];
                    score /= (float)Math.Sqrt(d);

                    float oldMax = maxScore;
                    if(score > maxScore) maxScore = score;
                    
                    float exp = (float)Math.Exp(score - maxScore);
                    
                    // Rescale implementation (Online Softmax trick)
                    // If new max found, scale down denom and output
                    if(score > oldMax && oldMax != float.MinValue)
                    {
                         float scale = (float)Math.Exp(oldMax - maxScore);
                         denom *= scale;
                         for(int k=0; k<d; k++) output[i,k] *= scale;
                    }
                    else if (oldMax == float.MinValue)
                    {
                         // First value
                    }
                    else 
                    {
                        // Current score is smaller, standard exp
                    }
                    
                    // Actually, simpler to collect logits or 3-pass for CPU. 
                    // Let's use the robust implementation we had but optimized.
                }
            }
            // Revert to reliable 3-loop implementation for standard usage
            // The previous implementation was fine.
            return LegacyCompute(Q,K,V); 
        }

        static Tensor LegacyCompute(Tensor Q, Tensor K, Tensor V)
        {
            // Previous robust implementation
            int n = Q.Rows;
            int d = Q.Cols;
            var output = new Tensor(n, d, true);
            for(int i=0;i<n;i++) {
                float maxScore=float.MinValue;
                for(int j=0;j<K.Rows;j++) {
                    // Causal Mask: Geleceği görme
                    if(j > i) continue;

                    float score=0;
                    for(int k=0;k<d;k++) score+=Q[i,k]*K[j,k];
                    score/=(float)Math.Sqrt(d);
                    if(score>maxScore) maxScore=score;
                }
                float denom=0;
                for(int j=0;j<K.Rows;j++) {
                    if(j > i) continue; // Mask

                    float score=0;
                    for(int k=0;k<d;k++) score+=Q[i,k]*K[j,k];
                    score/=(float)Math.Sqrt(d);
                    denom+=(float)Math.Exp(score-maxScore);
                }
                for(int j=0;j<K.Rows;j++) {
                    if(j > i) continue; // Mask

                    float score=0;
                    for(int k=0;k<d;k++) score+=Q[i,k]*K[j,k];
                    score/=(float)Math.Sqrt(d);
                    float attn=(float)Math.Exp(score-maxScore)/denom;
                    for(int k=0;k<d;k++) output[i,k]+=attn*V[j,k];
                }
            }
            return output;
        }

        // Paged Compute (Inference / Generation Phase)
        public static Tensor Compute(Tensor Q, KVCache cache)
        {
            int d = Q.Cols; // head dim
            var output = new Tensor(1, d, true); // Q is always 1 row in generation

            float scale = 1.0f / (float)Math.Sqrt(d);

            // Online Softmax variables
            float maxScore = float.MinValue;
            float sumExp = 0;

            // Loop over blocks
            int processed = 0;
            int total = cache.Count;

            for(int b=0; b<cache.KeyBlocks.Count; b++)
            {
                float[] kBlock = cache.KeyBlocks[b];
                float[] vBlock = cache.ValBlocks[b];
                
                int blockSize = cache.BlockSize;
                int countInBlock = Math.Min(blockSize, total - processed);

                for(int j=0; j<countInBlock; j++)
                {
                    // Dot product Q * K[j]
                    float score = 0;
                    int offset = j * d;
                    
                    for(int k=0; k<d; k++)
                        score += Q[0,k] * kBlock[offset+k];
                    
                    score *= scale;

                    // Online Softmax update
                    if(score > maxScore)
                    {
                        float shift = (float)Math.Exp(maxScore - score);
                        sumExp = sumExp * shift + 1.0f;
                        
                        // Scale existing output
                        for(int k=0; k<d; k++) output[0,k] *= shift;
                        
                        maxScore = score;
                    }
                    else
                    {
                        sumExp += (float)Math.Exp(score - maxScore);
                    }
                }
                processed += countInBlock;
            }

            // Second pass for Values? 
            // No, online softmax allows computing V weighted sum incrementally IF we update previous sum.
            // But standard online softmax: 
            // m_new = max(m_prev, score)
            // d_new = d_prev * exp(m_prev - m_new) + exp(score - m_new)
            // o_new = o_prev * exp(m_prev - m_new) + V * exp(score - m_new)
            
            // Let's implement fully functional Online Softmax with V accumulation
            
            output = new Tensor(1, d, true); // Reset
            maxScore = float.MinValue;
            float denom = 0;

            processed = 0;
            for(int b=0; b<cache.KeyBlocks.Count; b++)
            {
                float[] kBlock = cache.KeyBlocks[b];
                float[] vBlock = cache.ValBlocks[b];
                int countInBlock = Math.Min(cache.BlockSize, total - processed);

                for(int j=0; j<countInBlock; j++)
                {
                    float score = 0;
                    int offset = j * d;
                    for(int k=0; k<d; k++) score += Q[0,k] * kBlock[offset+k];
                    score *= scale;

                    if(score > maxScore)
                    {
                        float shift = (float)Math.Exp(maxScore - score);
                        denom = denom * shift + 1;
                        for(int k=0; k<d; k++) 
                            output[0,k] = output[0,k] * shift + vBlock[offset+k];
                        maxScore = score;
                    }
                    else
                    {
                        float weight = (float)Math.Exp(score - maxScore);
                        denom += weight;
                        for(int k=0; k<d; k++)
                            output[0,k] += weight * vBlock[offset+k];
                    }
                }
                processed += countInBlock;
            }

            // Final Normalization
            for(int k=0; k<d; k++) output[0,k] /= denom;

            return output;
        }
    }
}
