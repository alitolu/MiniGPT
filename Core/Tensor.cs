using System;

namespace MiniGPT.Core
{
    public class Tensor
    {
        public float[] Data;
        public float[] Grad;
        public Float16[] Data16;
        public bool UseFP16=false;

        public int Rows;
        public int Cols;

        public bool RequiresGrad;

        public Tensor(int r, int c, bool grad=false, bool useFP16=false)
        {
            Rows = r;
            Cols = c;
            RequiresGrad = grad;

            if(useFP16)
            {
                UseFP16=true;
                Data16=new Float16[r*c];
            }
            else
            {
                Data = new float[r*c];
            }

            if (grad) Grad = new float[r*c];
        }

        static Random rnd = new();

        public static Tensor Rand(int r,int c,bool grad=false)
        {
            var t = new Tensor(r,c,grad);
            for(int i=0;i<t.Data.Length;i++)
                t.Data[i]=(float)(rnd.NextDouble()*0.02-0.01);
            return t;
        }

        public float this[int r,int c]
        {
            get
            {
                int i=r*Cols+c;
                return UseFP16 ? Data16[i].ToFloat() : Data[i];
            }
            set
            {
                int i=r*Cols+c;
                if(UseFP16)
                    Data16[i]=new Float16(value);
                else
                    Data[i]=value;
            }
        }

        public void ZeroGrad()
        {
            if (Grad == null) return;
            Array.Clear(Grad,0,Grad.Length);
        }
    }
}
