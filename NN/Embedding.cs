using MiniGPT.Core;
using System;
using System.Collections.Generic;

namespace MiniGPT.NN
{
    public class Embedding
    {
        public Tensor W; // (Vocab, Dim)
        public int Vocab, Dim;
        
        int[] lastInput; // For backward

        public Embedding(int v, int d)
        {
            Vocab=v; Dim=d;
            W = Tensor.Rand(v, d, true); 
        }

        public Tensor Forward(int[] indices)
        {
            lastInput = indices;
            var outT = new Tensor(indices.Length, Dim, true);
            
            // Sparse Lookup (Copy memory block)
            for(int i=0; i<indices.Length; i++)
            {
                int id = indices[i];
                if(id >= Vocab) continue; 
                
                // Copy row W[id] to outT[i]
                Array.Copy(W.Data, id * Dim, outT.Data, i * Dim, Dim);
            }
            return outT;
        }

        public void Backward(Tensor gradOutput)
        {
            // gradOutput: (Seq, Dim)
            // Sparse Update on W.Grad
            if(W.Grad == null) W.Grad = new float[W.Data.Length];

            // Accumulate gradients
            // Sequential loop to avoid race conditions on same token ID updates
            for(int i=0; i<gradOutput.Rows; i++)
            {
                int id = lastInput[i];
                if(id >= Vocab) continue;

                int offsetW = id * Dim;
                int offsetG = i * Dim;
                
                // Add gradOutput[i] to W.Grad[id]
                for(int k=0; k<Dim; k++)
                {
                    W.Grad[offsetW + k] += gradOutput.Data[offsetG + k];
                }
            }
        }
        
        public IEnumerable<Tensor> Parameters() { yield return W; }
        
        public void Quantize() 
        { 
             // Quantization logic for embedding if needed
             // Currently kept as float for training stability
        }
    }
}
