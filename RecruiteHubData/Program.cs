using Microsoft.ML;
using Microsoft.ML.Data;


Console.WriteLine(@"
  _____             _   _  _  __ _____  _   _   _____   __  __   ____   _____   ______  _      
 |  __ \     /\    | \ | || |/ /|_   _|| \ | | / ____| |  \/  | / __ \ |  __ \ |  ____|| |     
 | |__) |   /  \   |  \| || ' /   | |  |  \| || |  __  | \  / || |  | || |  | || |__   | |     
 |  _  /   / /\ \  | . ` ||  <    | |  | . ` || | |_ | | |\/| || |  | || |  | ||  __|  | |     
 | | \ \  / ____ \ | |\  || . \  _| |_ | |\  || |__| | | |  | || |__| || |__| || |____ | |____ 
 |_|  \_\/_/    \_\|_| \_||_|\_\|_____||_| \_| \_____| |_|  |_| \____/ |_____/ |______||______|
                                                                                               
===============================================================================================
===============================================================================================
");
const string AssetsPath = @"../../../Assets";

string InputPath = Path.Combine(AssetsPath, "Input");
string OutputPath = Path.Combine(AssetsPath, "Output");
string TrainDatasetPath = Path.Combine(InputPath, "candidates-train.csv");
string ValidationDatasetPath = Path.Combine(InputPath, "candidates-validation.csv");
string TestDatasetPath = Path.Combine(InputPath, "candidates-test.csv");
string ModelPath = Path.Combine(OutputPath, "RankingModel.zip");

const string FeaturesVectorName = "Features";

// Create a common ML.NET context.
// Seed set to any number so you have a deterministic environment for repeateable results.
MLContext mlContext = new MLContext(seed: 0);

// Create the pipeline using the training data's schema; the validation and testing data have the same schema.
IDataView trainData = mlContext.Data.LoadFromTextFile<RankingData>(TrainDatasetPath, hasHeader: true, separatorChar: ',');
IEstimator<ITransformer> pipeline = CreatePipeline(mlContext, trainData);


// Train the model on the training dataset. To perform training you need to call the Fit() method.
Console.WriteLine("===== Train the model on the training dataset =====\n");
ITransformer model = pipeline.Fit(trainData);

// Evaluate the model using the metrics from the validation dataset; you would then retrain and reevaluate the model until the desired metrics are achieved.
Console.WriteLine("===== Evaluate the model's result quality with the validation data =====\n");
IDataView validationData = mlContext.Data.LoadFromTextFile<RankingData>(ValidationDatasetPath, separatorChar: ',', hasHeader: true);
EvaluateModel(mlContext, model, validationData);


// Combine the training and validation datasets.
var validationDataEnum = mlContext.Data.CreateEnumerable<RankingData>(validationData, false);
var trainDataEnum = mlContext.Data.CreateEnumerable<RankingData>(trainData, false);
var trainValidationDataEnum = validationDataEnum.Concat<RankingData>(trainDataEnum);
IDataView trainValidationData = mlContext.Data.LoadFromEnumerable<RankingData>(trainValidationDataEnum);

// Train the model on the train + validation dataset.
Console.WriteLine("===== Train the model on the training + validation dataset =====\n");
model = pipeline.Fit(trainValidationData);


// Evaluate the model using the metrics from the testing dataset; you do this only once and these are your final metrics.
Console.WriteLine("===== Evaluate the model's result quality with the testing data =====\n");
IDataView testData = mlContext.Data.LoadFromTextFile<RankingData>(TestDatasetPath, separatorChar: ',', hasHeader: true);
EvaluateModel(mlContext, model, testData);


// Combine the training, validation, and testing datasets.
var testDataEnum = mlContext.Data.CreateEnumerable<RankingData>(testData, false);
var allDataEnum = trainValidationDataEnum.Concat(testDataEnum);
IDataView allData = mlContext.Data.LoadFromEnumerable(allDataEnum);

// Retrain the model on all of the data, train + validate + test.
Console.WriteLine("===== Train the model on the training + validation + test dataset =====\n");
model = pipeline.Fit(allData);

// Save and consume the model to perform predictions.
// Normally, you would use new incoming data; however, for the purposes of this sample, we'll reuse the test data to show how to do predictions.
ConsumeModel(mlContext, model, ModelPath, testData);

Console.WriteLine(@"
===============================================================================================
===============================================================================================");

void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data) 
{
    // Use the model to perform predictions on the test data.
    IDataView predictions = model.Transform(data);

    Console.WriteLine("===== Use metrics for the data using NDCG@3 =====\n");

    // Evaluate the metrics for the data using NDCG; by default, metrics for the up to 3 search results in the query are reported (e.g. NDCG@3).
    ConsoleHelper.EvaluateMetrics(mlContext, predictions);

    // Evaluate metrics for up to 10 search results (e.g. NDCG@10).
    // TO CHECK:
    //Console.WriteLine("===== Use metrics for the data using NDCG@10 =====\n");
    //ConsoleHelper.EvaluateMetrics(mlContext, predictions, 10);
}


IEstimator<ITransformer> CreatePipeline(MLContext mlContext, IDataView dataView)
{
    const string FeaturesVectorName = "Features";

    Console.WriteLine("===== Set up the trainer =====\n");

    // Create an Estimator and transform the data:
    // 1. Concatenate the feature columns into a single Features vector.
    // 2. Create a key type for the label input data by using the value to key transform.
    // 3. Create a key type for the group input data by using a hash transform.
    IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(RankingData.Label))
    .Append(mlContext.Transforms.Text.FeaturizeText("EducationFeaturized", nameof(RankingData.Education)))
    .Append(mlContext.Transforms.Text.FeaturizeText("SkillSetFeaturized", nameof(RankingData.SkillSet)))
    .Append(mlContext.Transforms.Concatenate(FeaturesVectorName, nameof(RankingData.YearsExperience), "EducationFeaturized", "SkillSetFeaturized"))
    .Append(mlContext.Transforms.Conversion.Hash(nameof(RankingData.GroupId), nameof(RankingData.GroupId), numberOfBits: 20));

    // Set the LightGBM LambdaRank trainer.
    IEstimator<ITransformer> trainer = mlContext.Ranking.Trainers.LightGbm(labelColumnName: nameof(RankingData.Label), featureColumnName: FeaturesVectorName, rowGroupColumnName: nameof(RankingData.GroupId),numberOfIterations:200000);
    IEstimator<ITransformer> trainerPipeline = dataPipeline.Append(trainer);

    return trainerPipeline;
}


void ConsumeModel(MLContext mlContext, ITransformer model, string modelPath, IDataView data)
{
    Console.WriteLine("===== Save the model =====\n");

    // Save the model
    mlContext.Model.Save(model, null, modelPath);

    Console.WriteLine("===== Consume the model =====\n");

    // Load the model to perform predictions with it.
    DataViewSchema predictionPipelineSchema;
    ITransformer predictionPipeline = mlContext.Model.Load(modelPath, out predictionPipelineSchema);

    // Predict rankings.
    IDataView predictions = predictionPipeline.Transform(data);

    // In the predictions, get the scores of the search results included in the first query (e.g. group).
    IEnumerable<RankingPrediction> searchQueries = mlContext.Data.CreateEnumerable<RankingPrediction>(predictions, reuseRowObject: false);
    var firstGroupId = searchQueries.First().GroupId;
    IEnumerable<RankingPrediction> firstGroupPredictions = searchQueries.Take(100).Where(p => p.GroupId == firstGroupId).OrderByDescending(p => p.Score).ToList();

    // The individual scores themselves are NOT a useful measure of result quality; instead, they are only useful as a relative measure to other scores in the group. 
    // The scores are used to determine the ranking where a higher score indicates a higher ranking versus another candidate result.
    ConsoleHelper.PrintScores(firstGroupPredictions);
}




internal class RankingData
{
    [LoadColumn(0)]
    public float Label { get; set; }
    [LoadColumn(1)]
    public float GroupId { get; set; }
    [LoadColumn(2)]
    public string Education { get; set; }
    [LoadColumn(3)]
    public float YearsExperience { get; set; }
    [LoadColumn(4)]
    public string FullName { get; set; }
    [LoadColumn(5)]
    public string SkillSet { get; set; }
}

internal class RankingPrediction
{
    public float Score { get; set; }
    public string FullName { get; set; }
    public string SkillSet { get; set; }
    public uint GroupId { get; set; }
}

internal class ConsoleHelper
{
    // To evaluate the accuracy of the model's predicted rankings, prints out the Discounted Cumulative Gain and Normalized Discounted Cumulative Gain for search queries.
    public static void EvaluateMetrics(MLContext mlContext, IDataView predictions)
    {
        // Evaluate the metrics for the data using NDCG; by default, metrics for the up to 3 search results in the query are reported (e.g. NDCG@3).
        RankingMetrics metrics = mlContext.Ranking.Evaluate(predictions);

        Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");

        Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
    }

    // Performs evaluation with the truncation level set up to 10 search results within a query.
    // This is a temporary workaround for this issue: https://github.com/dotnet/machinelearning/issues/2728.
    public static void EvaluateMetrics(MLContext mlContext, IDataView predictions, int truncationLevel)
    {
        if (truncationLevel < 1 || truncationLevel > 10)
        {
            throw new InvalidOperationException("Currently metrics are only supported for 1 to 10 truncation levels.");
        }

        //  Uses reflection to set the truncation level before calling evaluate.
        var mlAssembly = typeof(TextLoader).Assembly;
        var rankEvalType = mlAssembly.DefinedTypes.Where(t => t.Name.Contains("RankingEvaluator")).First();

        var evalArgsType = rankEvalType.GetNestedType("Arguments");
        var evalArgs = Activator.CreateInstance(rankEvalType.GetNestedType("Arguments"));

        var dcgLevel = evalArgsType.GetField("DcgTruncationLevel");
        dcgLevel.SetValue(evalArgs, truncationLevel);

        var ctor = rankEvalType.GetConstructors().First();
        var evaluator = ctor.Invoke(new object[] { mlContext, evalArgs });

        var evaluateMethod = rankEvalType.GetMethod("Evaluate");
        RankingMetrics metrics = (RankingMetrics)evaluateMethod.Invoke(evaluator, new object[] { predictions, "Label", "GroupId", "Score" });

        Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");

        Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
    }

    // Prints out the the individual scores used to determine the relative ranking.
    public static void PrintScores(IEnumerable<RankingPrediction> predictions)
    {
        foreach (var prediction in predictions)
        {
            Console.WriteLine($"GroupId: {prediction.GroupId}, Score: {prediction.Score}");
        }
    }
}