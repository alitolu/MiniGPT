using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using MiniGPT.Runtime;
using MiniGPT.Engine;
using System;
using System.IO;

namespace MiniGPT.Server
{
    public record ChatRequest(string prompt);

    public static class ApiServer
    {
        public static void Run(ChatEngine engine)
        {
            var builder = WebApplication.CreateBuilder();
            builder.Services.AddCors();
            var app = builder.Build();

            app.UseCors(c => c.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
            
            app.UseDefaultFiles();
            app.UseStaticFiles(); 

            var generator = new StreamingGenerator(engine);

            app.MapPost("/chat", (ChatRequest req) =>
            {
                var reply = engine.Generate(req.prompt);
                return new { reply };
            });

            app.MapGet("/stream", async (HttpContext ctx) =>
            {
                string prompt = ctx.Request.Query["prompt"];
                if(string.IsNullOrEmpty(prompt)) prompt = "Merhaba";

                ctx.Response.ContentType = "text/event-stream";
                
                await foreach(var token in generator.GenerateStream(prompt, 50))
                {
                    await ctx.Response.WriteAsync($"data: {token}\n\n");
                    await ctx.Response.Body.FlushAsync();
                }
                
                await ctx.Response.WriteAsync("data: [DONE]\n\n");
            });

            Console.WriteLine("Server running at http://localhost:5000");
            app.Run("http://localhost:5000");
        }
    }
}
