package org.njit.cloudcomputing;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class QualityPrediction {

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


			Dataset<Row> label =input.withColumnRenamed("quality","label");
			label.show(10,false);

			VectorAssembler assembler =new VectorAssembler()
					.setInputCols(new String[] {"fixed acidity","volatile acidity","citric acid","residual sugar"
							,"chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
					.setOutputCol("features");


			Dataset<Row> features= assembler.transform(label).select("label","features");

			features=features.withColumn("label",functions.col("label").cast(DataTypes.DoubleType));
			features.show(10,false);


			System.out.println("\n============================LinearRegressionModel Start============================");

			System.out.println("Liner Model:: "+args[2]+"/linearModel");
			LinearRegressionModel linearModel = LinearRegressionModel.load(args[2]+"/linearModel");
			System.out.println("LinearRegression sameModel parameters:\n" + linearModel.explainParams() + "\n");
			System.out.println("LinearRegression sameModel coefficients :\n" + linearModel.coefficients() + "\n");
			System.out.println("LinearRegression sameModel intercept :\n" + linearModel.intercept() + "\n");
			//System.out.println("LinearRegression sameModel coefficients :\n" + sameModel.summary() + "\n");


			// Make predictions.
			Dataset<Row> predictions = linearModel.transform(features);
			// Select example rows to display.
			predictions.show(5,false);

			// Select example rows to display.
			predictions.select("label", "features").show(5);

			// Select (prediction, true label) and compute test error.
			RegressionEvaluator evaluator = new RegressionEvaluator()
					.setLabelCol("label")
					.setPredictionCol("prediction")
					.setMetricName("rmse");
			double rmse = evaluator.evaluate(predictions);
			System.out.println("Root Mean Squared Error (RMSE) / F1 Score:: " + rmse);




			System.out.println("\n============================LinearRegressionModel End============================");

			System.out.println("============================LogisticRegression Start============================\n");
			System.out.println("Logistic Model:: "+args[2]+"/logisticmodel");
			// Learn a LogisticRegression model. This uses the parameters stored in lr.
			LogisticRegressionModel logisticmodel = LogisticRegressionModel.load(args[2]+"/logisticmodel");
			// Since model1 is a Model (i.e., a Transformer produced by an Estimator),
			// we can view the parameters it used during fit().
			// This prints the parameter (name: value) pairs, where names are unique IDs for this
			// LogisticRegression instance.
			Dataset<Row> logisticPredictions = logisticmodel.transform(features);
			logisticPredictions.show(10,false);

			// Select (prediction, true label) and compute test error.
			RegressionEvaluator logisticEvaluator = new RegressionEvaluator()
					.setLabelCol("label")
					.setPredictionCol("prediction")
					.setMetricName("rmse");
			double rmse1 = logisticEvaluator.evaluate(logisticPredictions);
			System.out.println("Root Mean Squared Error (RMSE)  / F1 Score::  " + rmse1);

			System.out.println("============================LogisticRegression End============================\n");

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


