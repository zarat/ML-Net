ML.NET Beispiel-Daten

sentiment.csv  -> Label (true/false), Text
iris.csv       -> SepalLength, SepalWidth, PetalLength, PetalWidth, Label
house.csv      -> SizeM2, Bedrooms, AgeYears, PriceEur

var dv = ml.Data.LoadFromTextFile<T>(path, hasHeader: true, separatorChar: ',');
