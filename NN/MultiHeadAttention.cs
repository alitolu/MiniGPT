using MiniGPT.Core;
using MiniGPT.Engine;

namespace MiniGPT.NN
{
    public class MultiHeadAttention
    {
        Linear Wq,Wk,Wv,Wo;
        int dim;

        public MultiHeadAttention(int d)
        {
            dim=d;
            Wq=new Linear(d,d);
            Wk=new Linear(d,d);
            Wv=new Linear(d,d);
            Wo=new Linear(d,d);
        }

        public Tensor Forward(Tensor x, KVCache cache=null)
        {
            var Q=Wq.Forward(x);
            var K=Wk.Forward(x);
            var V=Wv.Forward(x);

            if(cache!=null)
            {
                cache.Add(K,V);

                K=cache.StackKeys();
                V=cache.StackValues();
            }

            var scores=new Tensor(Q.Rows,K.Rows,true);

            for(int i=0;i<Q.Rows;i++)
            for(int j=0;j<K.Rows;j++)
            {
                float s=0;
                for(int k=0;k<dim;k++)
                    s+=Q[i,k]*K[j,k];

                scores[i,j]=s/(float)System.Math.Sqrt(dim);
            }

            for(int i=0;i<scores.Rows;i++)
            {
                float sum=0;
                for(int j=0;j<scores.Cols;j++)
                {
                    scores[i,j]=(float)System.Math.Exp(scores[i,j]);
                    sum+=scores[i,j];
                }

                for(int j=0;j<scores.Cols;j++)
                    scores[i,j]/=sum;
            }

            var outv=new Tensor(Q.Rows,dim,true);

            for(int i=0;i<Q.Rows;i++)
            for(int j=0;j<K.Rows;j++)
            for(int k=0;k<dim;k++)
                outv[i,k]+=scores[i,j]*V[j,k];

            return Wo.Forward(outv);
        }
    }
}
