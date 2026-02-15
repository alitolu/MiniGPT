using MiniGPT.Model;
using MiniGPT.Engine;
using MiniGPT.Core;
using System.Collections.Generic;

namespace MiniGPT.Runtime
{
    public class InferenceEngine
    {
        MiniGPTModel model;
        KVCache[] caches;

        public InferenceEngine(MiniGPTModel model)
        {
            this.model = model;
            // Paged KVCache init
            caches = new KVCache[model.BlockCount];
            int headDim = model.Dim / model.Heads;
            for(int i=0; i<caches.Length; i++) 
                caches[i] = new KVCache(headDim);
        }

        public int NextToken(int[] context)
        {
            // Simple forward without history management for now
            // In real engine, we should maintain session state
            // Here we assume context is FULL history
            // But we can optimize to use cache if session is persistent
            
            // For now, stateless inference (re-implements ChatEngine logic partly)
            // But optimizes for "next token only"
            
            // If context is long, we should use cache prefill logic.
            // Since this Engine is stateless per request, we re-create cache?
            // No, InferenceEngine should be stateful or Session object needed.
            
            // For simplicity of Phase-11:
            // Input: FULL context.
            // Output: ONE next token.
            // Internally: Re-run model (slow) OR keep cache if user provides session ID.
            
            // Implementation: Fast "forward" only 
            var logits = model.Forward(context);
            // No cache used in this stateless call for simplicity, or we use cache locally
            // Using cache locally for single next token doesn't help if we don't persist it.
            
            return ArgMax(logits);
        }
        
        // Stateful Generator
        public IEnumerable<int> Stream(int[] prompt, int maxTokens)
        {
             throw new System.NotImplementedException("Use ChatEngine for now");
        }

        int ArgMax(float[] x)
        {
            int id=0;
            float m=x[0];
            for(int i=1;i<x.Length;i++) if(x[i]>m){m=x[i];id=i;}
            return id;
        }
    }
}
