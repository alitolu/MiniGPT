using MiniGPT.Core;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Optim
{
    public class AdamW
    {
        List<Tensor> p;
        float lr;

        public AdamW(IEnumerable<Tensor> parameters,float lr=3e-4f)
        {
            p=parameters.ToList();
            this.lr=lr;
        }

        public void Step()
        {
            foreach(var t in p)
            {
                for(int i=0;i<t.Data.Length;i++)
                    t.Data[i]-=lr*t.Grad[i];

                t.ZeroGrad();
            }
        }
    }
}
