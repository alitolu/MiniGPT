using System;

namespace MiniGPT.Embeddings
{
    public class EmbeddingModel
    {
        public float[] Embed(string text)
        {
            var vec = new float[128];

            for(int i=0;i<text.Length;i++)
                vec[i%128]+=text[i];

            Normalize(vec);
            return vec;
        }

        void Normalize(float[] v)
        {
            float sum=0;
            foreach(var x in v) sum+=x*x;

            float norm=(float)Math.Sqrt(sum);
            if(norm==0) norm=1;

            for(int i=0;i<v.Length;i++)
                v[i]/=norm;
        }
    }
}
