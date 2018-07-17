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

import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tree.impl.RandomForest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;

/**
 * TODO: add description.
 */
public class Main {

    public static void main(String[] argsList){
        SparkConf sparkConf = new SparkConf().setAppName("JavaRandomForestClassificationExample").setMaster("local");

        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
// Load and parse the data file.
        String datapath = "/home/ybabak/Downloads/homecredit_all.csv";
        SparkSession sparkSession = SparkSession.builder().appName("JavaRandomForestClassificationExample").getOrCreate();
        final Dataset<Row> ds = sparkSession.read().option("header", true).option("inferSchema", "true").csv(datapath);

        Dataset<Row>[] datasets = ds.randomSplit(new double[] {0.6, 0.4});

        Dataset<Row> trainDataset = datasets[0];
        trainDataset.cache();

        Dataset<Row> testDataset = datasets[1];
        testDataset.cache();

        ds.printSchema();


        // Step - 1: Make Vectors from dataframe's columns using special Vector Assmebler
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"REG_REGION_NOT_LIVE_REGION"})
            .setOutputCol("features");

        Dataset<Row> trainSet = assembler.transform(trainDataset);
        Dataset<Row> testSet = assembler.transform(testDataset);

        RandomForestClassifier trainer = new RandomForestClassifier()
            .setLabelCol("TARGET");
        RandomForestClassificationModel model = trainer.fit(trainSet);

        Dataset<Row> output = model.transform(testSet);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("TARGET")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(output);
        System.out.println(accuracy);
    }
}
