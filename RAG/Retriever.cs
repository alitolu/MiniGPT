using MiniGPT.Embeddings;

namespace MiniGPT.RAG
{
    public class Retriever
    {
        EmbeddingModel embed;
        VectorStore store;

        public Retriever(EmbeddingModel e, VectorStore s)
        {
            embed=e;
            store=s;
        }

        public string Retrieve(string query)
        {
            var q = embed.Embed(query);
            return store.Search(q);
        }
    }
}
