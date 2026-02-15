using System;

namespace MiniGPT.Core
{
    public class Q4Tensor
    {
        public byte[] Data; // 2 weights per byte
        public float[] Scales; // Block-wise scales (e.g. per 32 weights)
        public int Rows, Cols;
        int BlockSize=32;

        public Q4Tensor(Tensor t)
        {
            Rows=t.Rows;
            Cols=t.Cols;

            int total=Rows*Cols;
            Data=new byte[(total+1)/2];
            Scales=new float[(total+BlockSize-1)/BlockSize];

            for(int i=0;i<total;i+=BlockSize)
            {
                // Find scale for block
                float max=0;
                int end=Math.Min(i+BlockSize, total);
                for(int j=i;j<end;j++)
                {
                    float val=Math.Abs(t.Data[j]);
                    if(val>max) max=val;
                }

                float scale=max/7.0f; // 4-bit signed: -7..7
                Scales[i/BlockSize]=scale;
                if(scale==0) scale=1;

                // Quantize block
                for(int j=i;j<end;j++)
                {
                    float v=t.Data[j]/scale;
                    sbyte q=(sbyte)Math.Round(v);
                    if(q>7) q=7;
                    if(q<-7) q=-7;

                    // Pack into byte
                    // even index: low 4 bits, odd index: high 4 bits
                    int byteIdx=j/2;
                    bool high=(j%2)==1;

                    byte mask = (byte)(q & 0x0F);
                    if(high)
                        Data[byteIdx] |= (byte)(mask << 4);
                    else
                        Data[byteIdx] |= mask;
                }
            }
        }

        public float Get(int i)
        {
            int block=i/BlockSize;
            float scale=Scales[block];

            int byteIdx=i/2;
            bool high=(i%2)==1;

            byte b=Data[byteIdx];
            int val = high ? (b >> 4) : (b & 0x0F);
            
            // Sign extend 4-bit to 32-bit int
            if ((val & 0x08) != 0) val |= -16; // 0xFFFFFFF0

            return val*scale;
        }
    }
}
