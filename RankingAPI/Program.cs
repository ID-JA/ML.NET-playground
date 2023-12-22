using Microsoft.ML.Data;
using Microsoft.Extensions.ML;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile(modelName: "SentimentAnalysisModel", filePath: "./RankingModel.zip", watchForChanges: true);

var app = builder.Build();

app.UseHttpsRedirection();

var predictionHandler =
    async (PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool, ModelInput input) =>
        await Task.FromResult(predictionEnginePool.Predict(modelName: "SentimentAnalysisModel", input));


// Endpoint for making predictions
app.MapPost("/predict", async (PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool, ModelInput input) =>
{
    try
    {
        // Make a prediction using the specified model name
        var prediction = await Task.FromResult(predictionEnginePool.Predict(modelName: "SentimentAnalysisModel", input));

        // Return the prediction as JSON
        return Results.Json(prediction);
    }
    catch (Exception ex)
    {
        // Handle prediction error
        return Results.BadRequest($"Prediction failed: {ex.Message}");
    }
});


app.Run();

internal class ModelOutput
{
    [ColumnName("Score")]
    public float Score { get; set; }
}


internal class ModelInput
{
    public string Education { get; set; }
    public float YearsExperience { get; set; }
    public string FullName { get; set; }
    public string SkillSet { get; set; }
    public int GroupId { get; set; }
    public float Label { get; set; }
}