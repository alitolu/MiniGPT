using MiniGPT.Core;
using MiniGPT.Model;
using MiniGPT.Optim;

namespace MiniGPT.Engine
{
    public class Trainer
    {
        MiniGPTModel model;
        AdamW optim;

        public Trainer(MiniGPTModel m)
        {
            model=m;
            optim=new AdamW(model.Parameters());
        }

        public void TrainStep()
        {
            var x=Tensor.Rand(1,8);
            var y=new Tensor(1,8);

            var pred=model.Forward(x);

            for(int i=0;i<8;i++)
                pred.Grad[i]=pred.Data[i]-y.Data[i];

            optim.Step();
        }
    }
}
