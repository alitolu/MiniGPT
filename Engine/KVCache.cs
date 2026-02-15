using MiniGPT.Core;
using System.Collections.Generic;

namespace MiniGPT.Engine
{
    public class KVCache
    {
        public List<Tensor> Keys = new();
        public List<Tensor> Values = new();

        public void Add(Tensor k, Tensor v)
        {
            Keys.Add(k);
            Values.Add(v);
        }

        public Tensor StackKeys()
        {
            int rows = Keys.Count;
            int dim = Keys[0].Cols;

            var t = new Tensor(rows, dim);

            for(int i=0;i<rows;i++)
                for(int j=0;j<dim;j++)
                    t[i,j]=Keys[i][0,j];

            return t;
        }

        public Tensor StackValues()
        {
            int rows = Values.Count;
            int dim = Values[0].Cols;

            var t = new Tensor(rows, dim);

            for(int i=0;i<rows;i++)
                for(int j=0;j<dim;j++)
                    t[i,j]=Values[i][0,j];

            return t;
        }

        public void Clear()
        {
            Keys.Clear();
            Values.Clear();
        }
    }
}
