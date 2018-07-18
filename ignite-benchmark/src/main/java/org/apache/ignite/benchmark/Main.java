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

package org.apache.ignite.benchmark;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.lang.IgniteBiPredicate;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.environment.LearningEnvironment;
import org.apache.ignite.ml.environment.logging.ConsoleLogger;
import org.apache.ignite.ml.environment.logging.MLLogger;
import org.apache.ignite.ml.environment.parallelism.ParallelismStrategy;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.VectorUtils;
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator;
import org.apache.ignite.ml.selection.scoring.metric.Accuracy;
import org.apache.ignite.ml.selection.split.TrainTestDatasetSplitter;
import org.apache.ignite.ml.selection.split.TrainTestSplit;
import org.apache.ignite.ml.selection.split.mapper.SHA256UniformMapper;
import org.apache.ignite.ml.svm.SVMLinearBinaryClassificationTrainer;
import org.apache.ignite.ml.trainers.DatasetTrainer;
import org.apache.ignite.ml.tree.DecisionTreeClassificationTrainer;
import org.apache.ignite.ml.tree.boosting.GDBBinaryClassifierOnTreesTrainer;
import org.apache.ignite.ml.tree.randomforest.RandomForestClassifierTrainer;

/**
 * Running examples: nohup /usr/lib/jvm/java-8-oracle/bin/java -jar main.jar --dataset homecredit_top10k.csv
 * --cache-name HOMECREDIT --trainers svm,dt -p ignite -m server --config-path server.xml &> server.log &
 * /usr/lib/jvm/java-8-oracle/bin/java -jar main.jar --dataset homecredit_top10k.csv --cache-name HOMECREDIT --trainers
 * svm,dt -p ignite -m client --config-path client.xml
 */
public class Main {
    private static Set<String> algorithms = Stream.of("rf", "dt", "svm", "bst", "nn").collect(Collectors.toSet());
    private static final long COUNT_OF_ESTIMATIONS_PER_CASE = 3;

    public static class Args {
        /** Spring configuration path. */
        @Parameter(names = {"--config-path"}, required = true)
        public String configurationPath;

        /** Path to binary classification dataset in csv format. */
        @Parameter(names = {"--dataset", "-i"},
            required = true,
            description = "path to binary classification dataset in csv format")
        private String samplePath = "";

        /** Cache name. */
        @Parameter(names = {"--cache-name"})
        private String cacheName = "IGNITE_CACHE_NAME";

        /** Sample start size in percents of original sample size. */
        @Parameter(names = {"--sample-start-size"},
            description = "Sample start size in percents of original sample size")
        private double sampleStartSize = 0.1;

        /** Sample end size  in percents of original sample size. */
        @Parameter(names = {"--sample-end-size"},
            description = "Sample end size  in percents of original sample size")
        private double sampleEndSize = 1.0;

        /** Sample partition size will be increased by this value in each iteration. */
        @Parameter(names = {"--sample-size-step"},
            description = "Sample partition size will be increased by this value in each iteration")
        private double sampleSizeStep = 0.1;

        /** List of trainer names. */
        @Parameter(names = {"--trainers", "-t"},
            description = "list of values from set [rf, svm, dt, bst]")
        private List<String> trainers = Collections.singletonList("rf");

        /** Random seed. */
        @Parameter(names = {"--seed", "-s"}, description = "seed of random generator")
        private Long seed = System.currentTimeMillis();

        /** Output file prefix. */
        @Parameter(names = {"--output-file-prefix", "-p"},
            description = "prefix of output file names")
        private String outputFilePrefix = "ignite";

        /** Configuration name. */
        @Parameter(names = {"--configuration", "-c"},
            description = "Optional configuration name, it can has values from set: [min, max]. " +
                "\"min\" cofiguration corresponds to [sample-start-size=0.01, sample-end-size=0.1, step=0.01], " +
                "\"max\" cofiguration corresponds to [sample-start-size=0.1, sample-end-size=1.0, step=0.1]. " +
                "If configuration name is not set, then --sample-start-size, --sample-end-size, --sample-size-step will be used.",
            required = false)
        private String configurationName = "";

        @Parameter(names = {"--mode", "-m"}, description = "benchmark tool mode [client or server]", required = true)
        private String mode = "client";

        @Parameter(names = {"--help", "-h"}, help = true)
        private boolean help = false;
    }

    private static Ignite igniteInstance;

    public static void main(String[] argsList) {
        parseArgs(argsList).ifPresent(args -> {
            if (args.mode.equalsIgnoreCase("client"))
                startClient(args);
            else
                startServer(args);
        });
    }

    private static void startClient(Args args) {
        try (Ignite ignite = Ignition.start(args.configurationPath)) {
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>> CLIENT MODE");
            igniteInstance = ignite;
            tryDestroyCache(args, ignite);
            startBenchmark(args);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private static void startServer(Args args) {
        try {
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>> SERVER MODE");
            Ignition.start(args.configurationPath);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private static Optional<Args> parseArgs(String[] argsList) {
        Args args = new Args();
        JCommander jCommander = JCommander.newBuilder()
            .addObject(args).build();
        jCommander.parse(argsList);
        if (args.help) {
            jCommander.usage();
            return Optional.empty();
        }

        normalizeArgs(args);
        return Optional.of(args);
    }

    private static void normalizeArgs(Args args) {
        if (args.trainers.isEmpty())
            throw new IllegalArgumentException("Trainers list cannot be empty");
        args.trainers = args.trainers.stream().filter(x -> algorithms.contains(x)).collect(Collectors.toList());

        switch (args.configurationName) {
            case "":
                args.sampleStartSize = Math.max(0.0, Math.min(1.0, args.sampleStartSize));
                args.sampleEndSize = Math.min(1.0, Math.max(0.0, args.sampleEndSize));
                args.sampleSizeStep = Math.max(0.01, Math.min(0.99, args.sampleSizeStep));
                break;
            case "min":
                args.sampleStartSize = 0.01;
                args.sampleEndSize = 0.1;
                args.sampleSizeStep = 0.01;
                break;
            case "max":
                args.sampleStartSize = 0.1;
                args.sampleEndSize = 1.0;
                args.sampleSizeStep = 0.1;
                break;
        }
    }

    private static void tryDestroyCache(Args args, Ignite ignite) {
        try {
            ignite.destroyCache(args.cacheName);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static IgniteCache<Integer, VectorWithAswer> fillCache(Args args, Ignite ignite, double fraction, int sampleSize) {
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> CREATE AND FILL DATA_CACHE [fraction = %.4f]", fraction));

        CacheConfiguration<Integer, VectorWithAswer> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));
        cacheConfiguration.setName(args.cacheName);
        IgniteCache<Integer, VectorWithAswer> cache = ignite.createCache(cacheConfiguration);

        AtomicInteger counter = new AtomicInteger(0);
        try {
            Iterator<String> iter = Files.lines(Paths.get(args.samplePath)).iterator();
            while (iter.hasNext()) {
                String line = iter.next();
                if(counter.get() <= fraction * sampleSize)
                    cache.put(counter.getAndIncrement(), new VectorWithAswer(line));
                else
                    break;
            }
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        return cache;
    }

    private static void startBenchmark(Args args) {
        System.out.println(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK");
        int sampleSize = computeSampleSize(args.samplePath);
        for (String trainerName : args.trainers) {
            String outputFile = args.outputFilePrefix + "_" + trainerName + ".csv";
            printHeader(outputFile);

            DatasetTrainer<? extends Model<Vector, Double>, Double> trainer = createTrainer(trainerName);
            trainer.setEnvironment(LearningEnvironment.builder()
                .withParallelismStrategy(ParallelismStrategy.Type.ON_DEFAULT_POOL)
                .withLoggingFactory(ConsoleLogger.factory(MLLogger.VerboseLevel.MID))
                .build());

            for (double partSize = args.sampleStartSize;
                partSize <= args.sampleEndSize;
                partSize += args.sampleSizeStep) {

                long delta = 0L;
                double accuracy = 0.0;
                int retriesCount = 0;

                tryDestroyCache(args, igniteInstance);
                IgniteCache<Integer, VectorWithAswer> trainset = fillCache(args, igniteInstance, partSize, sampleSize);
                Exception lastException = null;
                for (int i = 0; i < COUNT_OF_ESTIMATIONS_PER_CASE; i++) {
                    if (retriesCount > 5)
                        throw new RuntimeException("Retries limit has exceeded", lastException);

                    try {
                        EstimationPair estimation = estimateModel(trainer, trainset, partSize, args);
                        delta += estimation.time;
                        accuracy += estimation.accuracy;
                    }
                    catch (RuntimeException e) {
                        e.printStackTrace();
                        i--;
                        retriesCount++;
                        lastException = e;
                    }
                }
                delta /= COUNT_OF_ESTIMATIONS_PER_CASE;
                accuracy /= COUNT_OF_ESTIMATIONS_PER_CASE;

                printEstimationResult(outputFile, partSize, trainerName, delta, accuracy);
            }
        }

        System.out.println(">>>>>>>>>>>>>>>>>>>>>> DONE");
    }

    private static int computeSampleSize(String path) {
        try {
            return Files.lines(Paths.get(path)).skip(1).mapToInt(line -> 1).sum();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
            return -1;
        }
    }

    private static DatasetTrainer<? extends Model<Vector, Double>, Double> createTrainer(String name) {
        switch (name) {
            case "rf":
                return new RandomForestClassifierTrainer(
                    900, 30,
                    1000, 0.1, 3,
                    0.0
                );
            case "dt":
                return new DecisionTreeClassificationTrainer(10, 0.0);
            case "bst":
                return new GDBBinaryClassifierOnTreesTrainer(1.0, 500, 1, 0.0);
            case "svm":
                return new SVMLinearBinaryClassificationTrainer();
            default:
                throw new IllegalArgumentException(name);
        }
    }

    private static EstimationPair estimateModel(DatasetTrainer<? extends Model<Vector, Double>, Double> trainer,
        IgniteCache<Integer, VectorWithAswer> trainset,
        double size, Args args) {

        String trainerName = trainer.getClass().getSimpleName();
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START LEARNING of %s [part size = %.4f]", trainerName, size));

        final boolean isSvm = trainer instanceof SVMLinearBinaryClassificationTrainer;

        long startTime = System.currentTimeMillis();
        Model<Vector, Double> model = trainer.fit(
            igniteInstance,
            trainset,
            createFilter(size, args.seed, true),
            (k, v) -> v.features(),
            (k, v) -> isSvm ? (v.answer() * 2 - 1) : v.answer()
        );
        long timeDelta = System.currentTimeMillis() - startTime;

        double accuracy = Evaluator.evaluate(
            trainset,
            createFilter(size, args.seed, false),
            model,
            (k, v) -> v.features(),
            (k, v) -> isSvm ? (v.answer() * 2 - 1) : v.answer(),
            new Accuracy<>()
        );

        return new EstimationPair(timeDelta, accuracy);
    }

    private static IgniteBiPredicate<Integer, VectorWithAswer> createFilter(double size, long seed, boolean isTrain) {
        Random rnd = new Random(seed);
        SHA256UniformMapper<Integer, VectorWithAswer> mapper = new SHA256UniformMapper<>(rnd);
        TrainTestSplit<Integer, VectorWithAswer> split = new TrainTestDatasetSplitter<>(mapper)
            .split(0.667);

        IgniteBiPredicate<Integer, VectorWithAswer> splitFilter = isTrain ? split.getTrainFilter() : split.getTestFilter();
        return splitFilter::apply;
    }

    private static class EstimationPair {
        public final long time;
        public final double accuracy;

        public EstimationPair(long time, double accuracy) {
            this.time = time;
            this.accuracy = accuracy;
        }
    }

    private static void printHeader(String outputFilename) {
        try (FileOutputStream fos = new FileOutputStream(outputFilename, false);
             PrintWriter printer = new PrintWriter(fos)) {

            printer.println("trainer\ttime_delta_in_ms\tpart_size\taccuracy");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printEstimationResult(String outputFilename, double size, String trainerName, long timeDelta,
        double accuracy) {
        try (FileOutputStream fos = new FileOutputStream(outputFilename, true);
             PrintWriter printer = new PrintWriter(fos)) {

            printer.println(String.format("%s\t%d\t%.4f\t%.4f", trainerName, timeDelta, size, accuracy));
        }
        catch (IOException e) {
            System.out.println(String.format(
                "EXCEPTION: %s [delta time = %d ms on %s with size of set = %.4f]",
                e.getMessage(),
                timeDelta,
                trainerName,
                size)
            );
        }
    }

    public static class VectorWithAswer {
        private final double[] features;
        private final double answer;

        public VectorWithAswer(String line) {
            String[] split = line.split(",");
            answer = Double.valueOf(split[split.length - 1]);

            features = new double[split.length - 1];
            for (int i = 0; i < features.length; i++)
                features[i] = Double.valueOf(split[i]);
        }

        public VectorWithAswer(double[] features, double answer) {
            this.features = features;
            this.answer = answer;
        }

        public Vector features() {
            return VectorUtils.of(features);
        }

        public double answer() {
            return answer;
        }
    }
}
