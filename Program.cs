using MiniGPT.Tokenizer;
using MiniGPT.Model;
using MiniGPT.Engine;

class Program
{
    static void Main()
    {
        string corpus =
            "merhaba nasılsın iyiyim teşekkür ederim " +
            "yapay zeka öğreniyorum mini gpt yazıyoruz";

        var tokenizer=new BPETokenizer();
        tokenizer.Build(corpus);

        var model=new MiniGPTModel(tokenizer.VocabSize,32);

        var trainer=new Trainer(model);

        for(int i=0;i<500;i++)
            trainer.TrainStep();

        var chat=new ChatEngine(tokenizer,model,tokenizer.VocabSize);

        while(true)
        {
            Console.Write("Sen: ");
            var input=Console.ReadLine();

            var reply=chat.Generate(input,10);
            Console.WriteLine("MiniGPT: "+reply);
        }
    }
}

