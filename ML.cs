using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLDemo;

internal class Program
{
    private static readonly string DataDir = Path.Combine(AppContext.BaseDirectory, "data");
    private static readonly string ModelsDir = Path.Combine(AppContext.BaseDirectory, "models");

    private static readonly string SentimentPath = Path.Combine(DataDir, "sentiment.csv");
    private static readonly string IrisPath = Path.Combine(DataDir, "iris.csv");
    private static readonly string HousePath = Path.Combine(DataDir, "house.csv");

    private static readonly string SentimentModelPath = Path.Combine(ModelsDir, "sentiment.zip");
    private static readonly string IrisModelPath = Path.Combine(ModelsDir, "iris.zip");
    private static readonly string HouseModelPath = Path.Combine(ModelsDir, "house.zip");

    static void Main()
    {
        Directory.CreateDirectory(ModelsDir);

        var ml = new MLContext(seed: 1);

        while (true)
        {
            Console.WriteLine("\n==== ML.NET Demo ====");
            Console.WriteLine("1) Sentiment Analysis (Binary)");
            Console.WriteLine("2) Iris Species (Multiclass)");
            Console.WriteLine("3) Hauspreis (Regression)");
            Console.WriteLine("4) Modelle laden und testen");
            Console.WriteLine("0) Beenden");
            Console.Write("Auswahl: ");
            var input = Console.ReadLine()?.Trim();

            switch (input)
            {
                case "1": TrainSentiment(ml); break;
                case "2": TrainIris(ml); break;
                case "3": TrainHouse(ml); break;
                case "4": LoadAndPredict(ml); break;
                case "0": return;
                default: Console.WriteLine("Ungültige Auswahl."); break;
            }
        }
    }

    // ------------------------------------------------------
    // 1) Sentiment Analysis
    // ------------------------------------------------------
    private static void TrainSentiment(MLContext ml)
    {

        Console.WriteLine("== Sentiment Analysis ==");

        var data = ml.Data.LoadFromTextFile<SentimentData>(SentimentPath, hasHeader: true, separatorChar: ',');
        var split = ml.Data.TrainTestSplit(data, testFraction: 0.25);
        var pipeline = ml.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text)).Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression( labelColumnName: nameof(SentimentData.Label),  featureColumnName: "Features"));
        var model = pipeline.Fit(split.TrainSet);
        var pred = model.Transform(split.TestSet);

        var metrics = ml.BinaryClassification.Evaluate(pred, nameof(SentimentData.Label));
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");

        ml.Model.Save(model, split.TrainSet.Schema, SentimentModelPath);
        Console.WriteLine($"Gespeichert: {SentimentModelPath}");

        var engine = ml.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
        var sample = new SentimentData { Text = "This product is amazing and works perfectly!" };
        var result = engine.Predict(sample);
        Console.WriteLine($"Text: {sample.Text}");
        Console.WriteLine($"→ {(result.PredictedLabel ? "Positiv" : "Negativ")} (P={result.Probability:0.###})");

    }

    public class SentimentData
    {
        [LoadColumn(0)] public bool Label { get; set; }
        [LoadColumn(1)] public string Text { get; set; } = "";
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    // ------------------------------------------------------
    // 2) Iris Classification
    // ------------------------------------------------------
    private static void TrainIris(MLContext ml)
    {
        Console.WriteLine("== Iris Classification ==");

        var data = ml.Data.LoadFromTextFile<IrisData>(IrisPath, hasHeader: true, separatorChar: ',');

        var split = ml.Data.TrainTestSplit(data, testFraction: 0.25);

        // Wichtig: PredictedLabel (Key) NICHT überschreiben!
        // Stattdessen extra Text-Spalte PredictedLabelValue erzeugen.
        
        // hard
        
        var pipeline =
            ml.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: nameof(IrisData.Label))
              .Append(ml.Transforms.Concatenate("Features",
                  nameof(IrisData.SepalLength),
                  nameof(IrisData.SepalWidth),
                  nameof(IrisData.PetalLength),
                  nameof(IrisData.PetalWidth)))
              .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                  labelColumnName: "LabelKey",
                  featureColumnName: "Features"))
              .Append(ml.Transforms.Conversion.MapKeyToValue(
                  outputColumnName: "PredictedLabelValue",
                  inputColumnName: "PredictedLabel"));


        // easier
        /*
        var pipeline = ml.Transforms.Conversion.MapValueToKey("LabelKey", nameof(IrisData.Label)).Append(ml.Transforms.Concatenate("Features",
              nameof(IrisData.SepalLength),
              nameof(IrisData.SepalWidth),
              nameof(IrisData.PetalLength),
              nameof(IrisData.PetalWidth))).Append(ml.MulticlassClassification.Trainers.LightGbm(
              labelColumnName: "LabelKey",
              featureColumnName: "Features",
              numberOfIterations: 50)).Append(ml.Transforms.Conversion.MapKeyToValue(
              "PredictedLabelValue", "PredictedLabel"));
        */

        // bad/ no prediction
        /*
        var pipeline = ml.Transforms.Conversion.MapValueToKey("LabelKey", nameof(IrisData.Label)).Append(ml.Transforms.Concatenate("Features",
              nameof(IrisData.SepalLength),
              nameof(IrisData.SepalWidth),
              nameof(IrisData.PetalLength),
              nameof(IrisData.PetalWidth))).Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "LabelKey",
                featureColumnName: "Features",
                maximumNumberOfIterations: 50));
        */

        //var model = pipeline.Fit(split.TrainSet);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine("Training läuft...");
        var model = pipeline.Fit(split.TrainSet);
        sw.Stop();
        Console.WriteLine($"Training fertig nach {sw.Elapsed}.");


        var pred = model.Transform(split.TestSet);

        // Evaluate nutzt PredictedLabel als Key (so wie vom Trainer erzeugt)
        var metrics = ml.MulticlassClassification.Evaluate(
            pred,
            labelColumnName: "LabelKey",
            predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:0.###}");

        ml.Model.Save(model, split.TrainSet.Schema, IrisModelPath);
        Console.WriteLine($"Gespeichert: {IrisModelPath}");

        var engine = ml.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

        var sample = new IrisData
        {
            SepalLength = 6.1f,
            SepalWidth = 2.8f,
            PetalLength = 4.7f,
            PetalWidth = 1.2f
        };

        var result = engine.Predict(sample);
        Console.WriteLine($"Vorhersage: {result.PredictedLabelValue}");
    }


    public class IrisData
    {
        [LoadColumn(0)] public float SepalLength { get; set; }
        [LoadColumn(1)] public float SepalWidth { get; set; }
        [LoadColumn(2)] public float PetalLength { get; set; }
        [LoadColumn(3)] public float PetalWidth { get; set; }
        [LoadColumn(4)] public string Label { get; set; } = "";
    }

    public class IrisPrediction
    {
        // Wir lesen die Text-Spalte, die wir zusätzlich erzeugt haben:
        [ColumnName("PredictedLabelValue")]
        public string PredictedLabelValue { get; set; } = "";

        // Optional hilfreich zum Debuggen:
        public float[] Score { get; set; } = Array.Empty<float>();
    }

    // ------------------------------------------------------
    // 3) Regression (Hauspreis)
    // ------------------------------------------------------
    private static void TrainHouse(MLContext ml)
    {
        Console.WriteLine("== Hauspreis-Regression ==");

        var data = ml.Data.LoadFromTextFile<HouseData>(HousePath, hasHeader: true, separatorChar: ',');
        var split = ml.Data.TrainTestSplit(data, testFraction: 0.25);

        var pipeline = ml.Transforms.Concatenate("Features", nameof(HouseData.SizeM2),
                nameof(HouseData.Bedrooms), nameof(HouseData.AgeYears))
            .Append(ml.Regression.Trainers.FastTree(labelColumnName: nameof(HouseData.PriceEur),
                featureColumnName: "Features"));

        var model = pipeline.Fit(split.TrainSet);
        var pred = model.Transform(split.TestSet);
        var metrics = ml.Regression.Evaluate(pred, labelColumnName: nameof(HouseData.PriceEur));

        Console.WriteLine($"R²: {metrics.RSquared:0.###}, RMSE: {metrics.RootMeanSquaredError:0.###}");
        ml.Model.Save(model, split.TrainSet.Schema, HouseModelPath);
        Console.WriteLine($"Gespeichert: {HouseModelPath}");

        var engine = ml.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
        var sample = new HouseData { SizeM2 = 100, Bedrooms = 3, AgeYears = 10 };
        var result = engine.Predict(sample);
        Console.WriteLine($"Vorhersage für 100m², 3 Zimmer, 10 Jahre: {result.Score:0} EUR");
    }

    public class HouseData
    {
        [LoadColumn(0)] public float SizeM2 { get; set; }
        [LoadColumn(1)] public float Bedrooms { get; set; }
        [LoadColumn(2)] public float AgeYears { get; set; }
        [LoadColumn(3)] public float PriceEur { get; set; }
    }

    public class HousePrediction
    {
        [ColumnName("Score")] public float Score { get; set; }
    }

    // ------------------------------------------------------
    // 4) Modelle laden und testen
    // ------------------------------------------------------
    private static void LoadAndPredict(MLContext ml)
    {
        Console.WriteLine("== Modelle laden ==");

        if (File.Exists(SentimentModelPath))
        {
            var model = ml.Model.Load(SentimentModelPath, out _);
            var engine = ml.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var p = engine.Predict(new SentimentData { Text = "I dont know maybe another time." });
            Console.WriteLine($"Sentiment: {(p.PredictedLabel ? "Positiv" : "Negativ")}");
        }

        if (File.Exists(IrisModelPath))
        {
            var model = ml.Model.Load(IrisModelPath, out _);
            var engine = ml.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
            var p = engine.Predict(new IrisData { SepalLength = 6.3f, SepalWidth = 3.0f, PetalLength = 5.8f, PetalWidth = 2.1f });
            Console.WriteLine($"Iris: {p.PredictedLabelValue}");
        }

        if (File.Exists(HouseModelPath))
        {
            var model = ml.Model.Load(HouseModelPath, out _);
            var engine = ml.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
            var p = engine.Predict(new HouseData { SizeM2 = 120, Bedrooms = 4, AgeYears = 5 });
            Console.WriteLine($"Hauspreis: {p.Score:0} EUR");
        }
    }

}
