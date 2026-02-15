using System;

namespace MiniGPT.Core
{
    public class QuantizedTensor
    {
        public sbyte[] Data;
        public float Scale;
        public int Rows,Cols;

        public QuantizedTensor(Tensor t)
        {
            Rows=t.Rows;
            Cols=t.Cols;

            Data=new sbyte[Rows*Cols];

            float max=0;

            foreach(var v in t.Data)
                if(Math.Abs(v)>max) max=Math.Abs(v);

            Scale=max/127f;

            for(int i=0;i<Data.Length;i++)
                Data[i]=(sbyte)(t.Data[i]/Scale);
        }

        public float Get(int i)
        {
            return Data[i]*Scale;
        }
    }
}
