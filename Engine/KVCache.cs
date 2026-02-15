using MiniGPT.Core;
using System.Collections.Generic;
using System;

namespace MiniGPT.Engine
{
    public class KVCache
    {
        public List<float[]> KeyBlocks = new();
        public List<float[]> ValBlocks = new();
        
        public int BlockSize = 32;
        public int Dim;
        public int Count = 0;
        public int Capacity = 0;

        public int MaxTokens = 2048; // Sliding window limit

        public KVCache(int dim)
        {
            Dim = dim;
        }

        public void Add(Tensor k, Tensor v)
        {
            int rows = k.Rows;
            
            for(int i=0; i<rows; i++)
            {
                if (Count == Capacity)
                {
                    KeyBlocks.Add(new float[BlockSize * Dim]);
                    ValBlocks.Add(new float[BlockSize * Dim]);
                    Capacity += BlockSize;
                }

                int blockIdx = Count / BlockSize;
                int offset = (Count % BlockSize) * Dim;

                Array.Copy(k.Data, i*Dim, KeyBlocks[blockIdx], offset, Dim);
                Array.Copy(v.Data, i*Dim, ValBlocks[blockIdx], offset, Dim);

                Count++;
            
                if (Count > MaxTokens)
                    SlidingWindow();
            }
        }

        private void SlidingWindow()
        {
             // Remove first block if full
             if(KeyBlocks.Count > 1 && (Count - MaxTokens) >= BlockSize)
             {
                 KeyBlocks.RemoveAt(0);
                 ValBlocks.RemoveAt(0);
                 Count -= BlockSize;
                 Capacity -= BlockSize;
             }
        }
    }
}
