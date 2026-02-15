using System.Collections.Generic;

namespace MiniGPT.Embeddings
{
    public class VectorStore
    {
        List<(float[],string)> data = new();

        public void Add(float[] vec,string text)
            => data.Add((vec,text));

        public string Search(float[] query)
        {
            float best=-1;
            string result="";

            foreach(var (v,t) in data)
            {
                float sim = Cosine(query,v);
                if(sim>best){best=sim;result=t;}
            }
            return result;
        }

        float Cosine(float[] a,float[] b)
        {
            float dot=0;
            for(int i=0;i<a.Length;i++)
                dot+=a[i]*b[i];
            return dot;
        }
    }
}
