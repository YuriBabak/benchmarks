/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.benchmark;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * TODO: add description.
 */
public class Main {
    /** Path to dataset. */
    private static String datapath = "/home/ybabak/Downloads/homecredit_with_column_numbers.csv";

    private static Map<String, ResultEntry> results;

    /** */
    public static void main(String[] argsList){
        SparkConf sparkConf = new SparkConf().setAppName("ClassificationBenchmark").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        SparkSession sparkSes = SparkSession.builder().appName("ClassificationBenchmark").getOrCreate();

        System.out.println(">>>>>>>>>>>>>>>>>>>>>> START DATA LOADING");
        long startTime = System.currentTimeMillis();

        final Dataset<Row> ds = sparkSes.read().option("header", true).option("inferSchema", "true").csv(datapath);
        Dataset<Row>[] datasets = ds.randomSplit(new double[] {0.6, 0.4});

        Dataset<Row> trainDataset = datasets[0];
        trainDataset.cache();

        Dataset<Row> testDataset = datasets[1];
        testDataset.cache();

        String[] featureNames = IntStream.rangeClosed(1, 901).<String>mapToObj(String::valueOf).toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureNames)
            .setOutputCol("features");

        Dataset<Row> trainSet = assembler.transform(trainDataset);
        Dataset<Row> testSet = assembler.transform(testDataset);

        long timeDelta = System.currentTimeMillis() - startTime;
        System.out.println(">>>>>>>>>>>>>>>>>>>>>> FINISHED DATA LOADING");
        System.out.println(">>>>>>>>>>>>>>>>>>>>>> TIME FOR LOADING: " + format(timeDelta));

        results = new HashMap<>();

//        rfBenchmark(trainSet, testSet);
//        dtBenchmark(trainSet, testSet);
        gdbBenchmark(trainSet, testSet);
        svmBenchmark(trainSet, testSet);

        System.out.println(">>>>>>>>>>>>>>>>>>>>>> RESULTS:");

        results.forEach((k,v) -> {
            System.out.println(k);
            System.out.println("time: " + format(v.millis));
            System.out.println("accuracy " + v.accuracy);
        });
    }

    /**
     * SVM benchmark.
     *
     * @param trainSet Train set.
     * @param testSet Test set.
     */
    private static void svmBenchmark(Dataset<Row> trainSet, Dataset<Row> testSet) {
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK [%s mode]", "SVM"));
        long startTime = System.currentTimeMillis();

        LinearSVC trainer = new LinearSVC()
            .setLabelCol("TARGET");

        LinearSVCModel mdl = trainer.fit(trainSet);

        long timeDelta = System.currentTimeMillis() - startTime;

        evaluateModel(mdl, timeDelta, testSet);
    }

    /**
     * Gradient boosting on trees benchmark.
     *
     * @param trainSet Train set.
     * @param testSet Test set.
     */
    private static void gdbBenchmark(Dataset<Row> trainSet, Dataset<Row> testSet) {
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK [%s mode]", "GDB"));
        long startTime = System.currentTimeMillis();

        GBTClassifier trainer = new GBTClassifier()
            .setLabelCol("TARGET")
            .setMaxDepth(1)
            .setStepSize(1.0d)
            .setMaxIter(500);

        GBTClassificationModel mdl = trainer.fit(trainSet);

        long timeDelta = System.currentTimeMillis() - startTime;

        evaluateModel(mdl, timeDelta, testSet);
    }

    /**
     * DecisionTree benchmark.
     *
     * @param trainSet Train set.
     * @param testSet Test set.
     */
    private static void dtBenchmark(Dataset<Row> trainSet, Dataset<Row> testSet) {
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK [%s mode]", "DecisionTree"));
        long startTime = System.currentTimeMillis();

        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
            .setLabelCol("TARGET")
            .setMaxDepth(10)
            .setMinInfoGain(0.0d);

        DecisionTreeClassificationModel mdl = trainer.fit(trainSet);

        long timeDelta = System.currentTimeMillis() - startTime;

        evaluateModel(mdl, timeDelta, testSet);
    }

    /**
     * RandomForest benchmark.
     *
     * @param trainSet Train set.
     * @param testSet Test set.
     */
    private static void rfBenchmark(Dataset<Row> trainSet, Dataset<Row> testSet) {
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK [%s mode]", "RandomForest"));
        long startTime = System.currentTimeMillis();

        RandomForestClassifier trainer = new RandomForestClassifier()
            .setLabelCol("TARGET")
            .setMaxDepth(3)
            .setNumTrees(1000)
            .setMinInfoGain(0.0d)
            .setSubsamplingRate(0.1)
            .setFeatureSubsetStrategy("sqrt");
        RandomForestClassificationModel mdl = trainer.fit(trainSet);

        long timeDelta = System.currentTimeMillis() - startTime;

        evaluateModel(mdl, timeDelta, testSet);
    }

    /**
     * Evaluate model and print result, time and accuracy.
     *
     * @param mdl Model.
     * @param millis Millis.
     * @param testSet Test set.
     */
    private static void evaluateModel(ClassificationModel mdl, long millis, Dataset<Row> testSet){
        Dataset<Row> output = mdl.transform(testSet);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("TARGET")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(output);

        System.out.println("time: " + format(millis));
        System.out.println("accuracy " + accuracy);

        results.put(mdl.getClass().getSimpleName(), new ResultEntry(millis, accuracy));
    }

    /**
     * Format time form long to readable format.
     *
     * @param millis Millis.
     */
    private static String format(long millis){
        return String.format("%02d:%02d:%02d",
            TimeUnit.MILLISECONDS.toHours(millis),
            TimeUnit.MILLISECONDS.toMinutes(millis) -
                TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(millis)), // The change is in this line
            TimeUnit.MILLISECONDS.toSeconds(millis) -
                TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(millis)));
    }

    private static class ResultEntry{
        private long millis;
        private double accuracy;

        public ResultEntry(long millis, double accuracy) {
            this.millis = millis;
            this.accuracy = accuracy;
        }
    }
}