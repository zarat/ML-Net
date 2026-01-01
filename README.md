# ML.NET

## 1) Grundprinzip von ML.NET

ML.NET arbeitet fast immer nach diesem Muster:

1. **Daten laden** → `IDataView`  
2. **Pipeline bauen** (Transforms + Trainer) → `IEstimator`  
3. **Trainieren** → `Fit(...)` ergibt ein **Model** (`ITransformer`)  
4. **Testen/Evaluieren** → `model.Transform(test)` + `Evaluate(...)`  
5. **Speichern/Laden** → `ml.Model.Save(...)` / `ml.Model.Load(...)`  
6. **Vorhersagen** → `PredictionEngine` (Single) oder `Transform` (Batch)

Wichtige Begriffe:
- `MLContext`: Einstiegspunkt, hält Seed/Random, Logging, etc.
- `IDataView`: Lazy-Data-View (wird on-demand gelesen)
- `Transforms`: Feature Engineering (Text→Features, Normalisierung, Typ-Konvertierung…)
- `Trainer`: Lernalgorithmus, der aus Features + Label ein Modell macht

---

## 2) Projekt-Setup

### NuGet (Beispiel)
- `Microsoft.ML`
- `Microsoft.ML.FastTree` (Regression mit FastTree)
- `Microsoft.ML.LightGbm` (für Multiclass)

Trainingsdaten liegen im Unterordner /data
```
/data
  sentiment.csv
  iris.csv
  house.csv
  email_spam.csv
```

---

## 3) Datenformate und Bedeutung der Spalten

### 3.1 sentiment.csv (Binary Classification)
**Ziel:** Positiv vs. Negativ  
**Spalten:**
- `Label` (bool): `true` = positiv, `false` = negativ
- `Text` (string): der Text (Review/Kommentar)

**Warum Text-Features nötig sind:** Trainer erwarten Zahlen. Text wird über `FeaturizeText` in einen numerischen Vektor umgewandelt.

**CSV:** Kommas im Text → `allowQuoting: true` und CSV korrekt quoten:
```csharp
var dv = ml.Data.LoadFromTextFile<SentimentData>(
    SentimentPath, hasHeader: true, separatorChar: ',', allowQuoting: true);
```

---

### 3.2 iris.csv (Multiclass Classification)
**Ziel:** 3 Klassen (`setosa`, `versicolor`, `virginica`)  
**Spalten:**
- `SepalLength`, `SepalWidth`, `PetalLength`, `PetalWidth` (float): Messwerte
- `Label` (string): Klassenname

**Warum `MapValueToKey`:** Viele Multiclass-Trainer erwarten Labels als **Key** (intern: 0..K-1).  
Darum:
- `Label` (string) → `LabelKey` (Key)
- Prediction liefert `PredictedLabel` als Key
- zusätzlich erzeugt `PredictedLabelValue` (Key → string), damit man es ausgeben kann

---

### 3.3 house.csv (Regression)
**Ziel:** Preis als Zahl vorhersagen  
**Spalten:**
- `SizeM2` (float): Fläche
- `Bedrooms` (float): Zimmer (als Zahl)
- `AgeYears` (float): Alter
- `PriceEur` (float): **Label** (Zielwert)

Regression bedeutet: Output ist eine **kontinuierliche Zahl** (kein Label).

---

### 3.4 email_spam.csv (Binary Classification mit Text + Zusatzfeatures)
**Ziel:** Spam (`true`) vs. Ham (`false`)  
**Spalten:**
- `IsSpam` (bool): **Label**
- `Subject` (string): Betreff
- `Body` (string): Inhalt
- `HasAttachment` (bool): ob ein Anhang vorhanden ist
- `AttachmentCount` (float): Anzahl Anhänge (oder 0/1)
- `NumLinks` (float): Link-Anzahl (z.B. zählt `http://`)
- `BodyLength` (float): Länge des Bodys

**Ganz wichtig:** Wenn Text Felder Kommas enthalten (z.B. `$3,000/week`), müssen sie in Quotes stehen **und** ML.NET muss Quoting erlauben:
```csharp
var dv = ml.Data.LoadFromTextFile<EmailSpamData>(
    EmailSpamPath, hasHeader: true, separatorChar: ',', allowQuoting: true);
```

---

## 4) Warum 0.5/0.5 kommen kann

Trainer können nur mit numerischen Features arbeiten. ML.NET nutzt dafür Text-Transforms.

### 4.1 `FeaturizeText`
```csharp
ml.Transforms.Text.FeaturizeText("BodyFeats", nameof(EmailSpamData.Body))
```
Ergebnis: `BodyFeats` ist ein **Vektor aus floats** (z.B. n‑grams / TF‑IDF ähnliche Signale).

### 4.2 Warum `Concatenate("TextAll", Subject, Body)` nicht funktioniert
`Concatenate` ist für **Feature-Vektoren** (Zahlen), nicht zum „Strings zusammenkleben“.  
Mit Strings entstehen u.U. **nicht sinnvolle/unerwartete Spalten**, und `FeaturizeText` kann dann „nichts lernen“ → Score=0 → Probability=0.5.

**Richtige Lösung:** Subject und Body **getrennt featurizen** und später Feature-Vektoren zusammenfügen:
```csharp
.Append(ml.Transforms.Text.FeaturizeText("SubjectFeats", nameof(EmailSpamData.Subject)))
.Append(ml.Transforms.Text.FeaturizeText("BodyFeats", nameof(EmailSpamData.Body)))
.Append(ml.Transforms.Concatenate("Features", "SubjectFeats", "BodyFeats", ...))
```

---

## 5) Features kombinieren: Bool + Zahlen + Text

### 5.1 Bool → float
Viele Trainer erwarten numerische Features. Bool wird zu 0/1:
```csharp
.Append(ml.Transforms.Conversion.ConvertType("HasAttachmentF",
    nameof(EmailSpamData.HasAttachment), outputKind: DataKind.Single))
```

### 5.2 Normalisierung numerischer Spalten (optional, oft hilfreich)
```csharp
.Append(ml.Transforms.NormalizeMeanVariance(nameof(EmailSpamData.NumLinks)))
```
Das bringt Features auf ähnliches Skalen-Niveau → Optimierer konvergiert stabiler.

### 5.3 Feature-Vektor
Am Ende muss alles in **eine** Spalte `Features`:
```csharp
.Append(ml.Transforms.Concatenate("Features",
    "SubjectFeats", "BodyFeats", "HasAttachmentF",
    nameof(EmailSpamData.AttachmentCount),
    nameof(EmailSpamData.NumLinks),
    nameof(EmailSpamData.BodyLength)))
```

---

## 6) Warum welcher Trainer?

### 6.1 Binary Classification (Sentiment, Spam)
**SDCA Logistic Regression** (`SdcaLogisticRegression`)
- Sehr gängig, schnell, gut für viele sparse Features (Text).
- Kann bei sehr kleinen Datensätzen manchmal „zickig“ sein (dann hilft Iterations-Limit oder LBFGS).

**LBFGS Logistic Regression** (`LbfgsLogisticRegression`)
- Stabiler Optimierer, oft gut bei kleineren Datensätzen.
- Im Spam Beispiel liefert LBFGS schnell plausible Ergebnisse.

**Faustregel:**  
- Text + viele Features: SDCA oder LBFGS  
- Wenn SDCA komisch wirkt: testweise LBFGS

### 6.2 Multiclass (Iris)
**SDCA Maximum Entropy** (`SdcaMaximumEntropy`)
- Klassischer linearer Multiclass-Classifier.
- Wenn Training „ewig“ dauert → Iterationen begrenzen oder LightGBM probieren.

**LightGBM Multiclass**
- Gradient Boosted Trees, oft sehr stark und schnell, besonders bei tabellarischen Daten.
- Extra Paket nötig: `Microsoft.ML.LightGbm`.

### 6.3 Regression (Hauspreis)
**FastTree Regression**
- Boosted Trees, gut für nichtlineare Zusammenhänge.
- Robust auf tabellarischen Features.

---

## 7) Evaluationsmetriken verstehen

### 7.1 Binary
- **Accuracy**: Anteil richtig klassifiziert (kann bei unbalancierten Daten täuschen)
- **AUC**: Wie gut Spam/Ham getrennt werden kann (0.5 = Zufall, 1.0 = perfekt)
- **F1**: Balance aus Precision/Recall, nützlich bei Spam

### 7.2 Multiclass
- **MicroAccuracy**: gewichtet nach Häufigkeit
- **MacroAccuracy**: Durchschnitt pro Klasse (jede Klasse gleich wichtig)
- **LogLoss**: kleiner ist besser

### 7.3 Regression
- **R²**: 1.0 = perfekt, 0 = „nicht besser als Mittelwert“
- **RMSE**: Fehlergröße in Ziel-Einheiten (hier: Euro)

---

## 8) Modell speichern/laden und Vorhersagen

### 8.1 Speichern
```csharp
ml.Model.Save(model, train.Schema, EmailSpamModelPath);
```

### 8.2 Laden
```csharp
var model = ml.Model.Load(EmailSpamModelPath, out _);
```

### 8.3 PredictionEngine (Single Prediction)
Gut für einzelne Vorhersagen, nicht für Massendaten:
```csharp
var engine = ml.Model.CreatePredictionEngine<EmailSpamData, EmailSpamPrediction>(model);
var p = engine.Predict(sample);
```

### 8.4 Batch Predictions (empfohlen bei vielen Daten)
```csharp
var scored = model.Transform(dataView);
```
Mit `CreateEnumerable<...>` alle Scores iterieren.

---

## 9) Spam-Klassifikation praxisnaher machen

### 9.1 Schwellenwert (Threshold) setzen
Standard ist 0.5. Konservativer wäre:
- Spam nur, wenn `Probability > 0.9`
- Sonst „unsicher“ oder „Inbox“

### 9.2 Feature-Ideen
- `FromDomainIsExternal` (bool)
- `ContainsMoneyWords` (bool)
- `NumUppercase`, `NumExclamation`
- `ContainsUnsubscribe`
- `HasReplyToMismatch`

Viele davon kann man als zusätzliche CSV-Spalten hinzufügen oder zur Laufzeit berechnen.
