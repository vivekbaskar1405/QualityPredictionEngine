package org.njit.cloudcomputing;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.linalg.Vectors;

public class QualityPrediction_old {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			String inputfile = args[0];
			//"/Users/vivekbaskar/eclipse-workspace/QualityPredictionEngine/TrainingDataset.csv";
			SparkSession spark = SparkSession.
					builder()
					.master(args[1])
					.appName("Wine_Quality_Prediction").getOrCreate();

			JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
			jsc.setLogLevel("ERROR");

			Dataset<Row> input = spark.read()
					.option("header", true)
					.option("inferSchema",true)
					.option("delimiter",",")
					.format("csv")
					.load(inputfile);

			input.show(10,false);

			input.printSchema();

			Dataset<Row> label =input.withColumnRenamed("quality","label");
			label.show(10,false);
			label.printSchema();

			VectorAssembler assembler =new VectorAssembler()
					.setInputCols(new String[] {"fixed acidity","volatile acidity","citric acid","residual sugar"
							,"chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
					.setOutputCol("features");


			Dataset<Row> features= assembler.transform(label).select("label","features");

			features.show(10,false);
			features=features.withColumn("label",functions.col("label").cast(DataTypes.DoubleType));
			features.printSchema();

			LinearRegression linear = new LinearRegression();

			linear
			.setMaxIter(200)
			.setRegParam(0.3)
			.setElasticNetParam(0.2);


			LinearRegressionModel linearModel= linear.fit(features);


			System.out.println("LinearRegression parameters:\n" + linearModel.explainParams() + "\n");
			System.out.println("LinearRegression coefficients :\n" + linearModel.coefficients() + "\n");
			System.out.println("LinearRegression intercept :\n" + linearModel.intercept() + "\n");
			System.out.println("LinearRegression coefficients :\n" + linearModel.summary() + "\n");

			LinearRegressionTrainingSummary linearSummary =	linearModel.summary();
			System.out.println("LinearRegressionSummary  explainedVariance :\n" + linearSummary.explainedVariance() + "\n");
			System.out.println("LinearRegressionSummary  totalIterations :\n" + linearSummary.totalIterations() + "\n");
			System.out.println("LinearRegressionSummary  degreesOfFreedom :\n" + linearSummary.degreesOfFreedom() + "\n");
			System.out.println("objectiveHistory: " + Vectors.dense(linearSummary.objectiveHistory()));
			System.out.println("RMSE: " + linearSummary.rootMeanSquaredError());
			System.out.println("r2: " + linearSummary.r2());

			Dataset<Row> predictions=linearSummary.predictions();
			predictions.show(10,false);

			Dataset<Row> residuals=linearSummary.residuals();
			residuals.show(10,false);

			double r2=linearSummary.r2();
			System.out.println("R2 Measure :: "+r2);

			linearModel.write().overwrite().save(args[2]+"/linearModel");

			LinearRegressionModel sameModel = LinearRegressionModel.load(args[2]+"/linearModel");
			System.out.println("============================");
			System.out.println("LinearRegression sameModel parameters:\n" + sameModel.explainParams() + "\n");
			System.out.println("LinearRegression sameModel coefficients :\n" + sameModel.coefficients() + "\n");
			System.out.println("LinearRegression sameModel intercept :\n" + sameModel.intercept() + "\n");
			//System.out.println("LinearRegression sameModel coefficients :\n" + sameModel.summary() + "\n");


			// Make predictions.
			Dataset<Row> predictions1 = sameModel.transform(features);

			// Select example rows to display.
			predictions1.show(5,false);

			// Select (prediction, true label) and compute test error
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
					.setLabelCol("label")
					.setPredictionCol("prediction")
					.setMetricName("accuracy");
			double accuracy = evaluator.evaluate(predictions);
			System.out.println("Test accuracy = " +accuracy);
			System.out.println("Test Error = " + (1.0 - accuracy));


			System.out.println("============================");

			// Get evaluation metrics.
			//MulticlassMetrics metrics = new MulticlassMetrics(features);
			//double accuracy = Double.valueOf(metrics.accuracy());
			//System.out.println("Accuracy = " + accuracy);





			// Create a LogisticRegression instance. This instance is an Estimator.
			LogisticRegression lr = new LogisticRegression();
			// Print out the parameters, documentation, and any default values.
			System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");

			// We may set parameters using setter methods.
			lr.setMaxIter(200)
			.setRegParam(0.3)
			.setThreshold(0.1)
			.setElasticNetParam(0.2);

			// Learn a LogisticRegression model. This uses the parameters stored in lr.
			LogisticRegressionModel logisticmodel = lr.fit(features);
			// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
			// we can view the parameters it used during fit().
			// This prints the parameter (name: value) pairs, where names are unique IDs for this
			// LogisticRegression instance.
			System.out.println("Model 1 was fit using parameters: " + logisticmodel.parent().extractParamMap());

			System.out.println("LogisticRegression parameters:\n" + logisticmodel.explainParams() + "\n");
			System.out.println("LogisticRegression coefficients :\n" + logisticmodel.coefficientMatrix() + "\n");
			//System.out.println("LogisticRegression intercept :\n" + logisticmodel.intercept() + "\n");
			System.out.println("LogisticRegression coefficients :\n" + logisticmodel.summary() + "\n");

			LogisticRegressionTrainingSummary LogisticSummary =	logisticmodel.summary();
			System.out.println("LogisticSummary  accuracy :\n" + LogisticSummary.accuracy() + "\n");
			System.out.println("LogisticSummary  totalIterations :\n" + LogisticSummary.totalIterations() + "\n");
			System.out.println("LogisticSummary  weightedFalsePositiveRate :\n" + LogisticSummary.weightedFalsePositiveRate() + "\n");

			Dataset<Row> logisticPredictions=LogisticSummary.predictions();
			logisticPredictions.show(10,false);

			double[] F1Measure=LogisticSummary.fMeasureByLabel();

			for(double f :F1Measure) {
				System.out.println("F1 Measure :: "+f);
			}

			double fMeasure = LogisticSummary.weightedFMeasure();
			System.out.println("\nF1 fMeasure :: "+fMeasure);

			logisticmodel.write().overwrite().save(args[2]+"/logisticmodel");

			LogisticRegressionModel sameLogisticModel = LogisticRegressionModel.load(args[2]+"/logisticmodel");

			System.out.println("============================");
			//System.out.println("Model 1 was fit using parameters: " + sameLogisticModel.parent().extractParamMap());

			System.out.println("LogisticRegression sameLogisticModel parameters:\n" + sameLogisticModel.explainParams() + "\n");
			System.out.println("LogisticRegression sameLogisticModel coefficients :\n" + sameLogisticModel.coefficientMatrix() + "\n");
			//System.out.println("LogisticRegression sameLogisticModel intercept :\n" + logisticmodel.intercept() + "\n");
			//System.out.println("LogisticRegression sameLogisticModel coefficients :\n" + sameLogisticModel.summary() + "\n");
			System.out.println("============================");






			spark.stop();
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(0);
		}

	}
}
//./bin/spark-submit  --class org.njit.cloudcomputing.QualityPrediction  --master "spark://Viveks-MacBook-Pro.local:7077"  /Users/vivekbaskar/eclipse-workspace/QualityPredictionEngine/target/QualityPredictionEngine-jar-with-dependencies.jar  /Users/vivekbaskar/eclipse-workspace/QualityPredictionEngine/TrainingDataset.csv spark://Viveks-MacBook-Pro.local:7077 Users/vivekbaskar/eclipse-workspace/QualityPredictionEngine


