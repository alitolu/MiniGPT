using MiniGPT.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MiniGPT.Optim
{
    public class AdamW
    {
        List<Tensor> parameters;
        float lr;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;

        Dictionary<Tensor, float[]> m = new();
        Dictionary<Tensor, float[]> v = new();

        int step = 0;

        public AdamW(IEnumerable<Tensor> parameters, float lr = 3e-4f)
        {
            this.parameters = parameters.ToList();
            this.lr = lr;

            foreach (var p in this.parameters)
            {
                m[p] = new float[p.Data.Length];
                v[p] = new float[p.Data.Length];
            }
        }

        public void Step()
        {
            step++;

            foreach (var p in parameters)
            {
                for (int i = 0; i < p.Data.Length; i++)
                {
                    float g = p.Grad[i];

                    m[p][i] = beta1 * m[p][i] + (1 - beta1) * g;
                    v[p][i] = beta2 * v[p][i] + (1 - beta2) * g * g;

                    float mHat = m[p][i] / (1 - MathF.Pow(beta1, step));
                    float vHat = v[p][i] / (1 - MathF.Pow(beta2, step));

                    p.Data[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                }
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in parameters)
                p.ZeroGrad();
        }
    }
}
