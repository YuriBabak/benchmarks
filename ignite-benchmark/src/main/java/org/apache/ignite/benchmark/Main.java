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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.configuration.IgniteConfiguration;
import org.apache.ignite.ml.Model;
import org.apache.ignite.ml.math.Vector;
import org.apache.ignite.ml.math.VectorUtils;
import org.apache.ignite.ml.selection.split.mapper.SHA256UniformMapper;
import org.apache.ignite.ml.svm.SVMLinearBinaryClassificationTrainer;
import org.apache.ignite.ml.trainers.DatasetTrainer;
import org.apache.ignite.ml.tree.DecisionTreeClassificationTrainer;
import org.apache.ignite.ml.tree.boosting.GDBBinaryClassifierOnTreesTrainer;
import org.apache.ignite.ml.tree.randomforest.RandomForestClassifierTrainer;
import org.apache.ignite.spi.discovery.tcp.TcpDiscoverySpi;
import org.apache.ignite.spi.discovery.tcp.ipfinder.vm.TcpDiscoveryVmIpFinder;

public class Main {
    private static Set<String> algorithms = Stream.of("rf", "dt", "svm", "bst", "nn").collect(Collectors.toSet());
    private static final long COUNT_OF_ESTIMATIONS_PER_CASE = 3;

    public static class Args {
        /** Sample path. */
        @Parameter(names = {"--dataset"}, description = "path to dataset in csv format")
        private String samplePath = "";

        /** Cache name. */
        @Parameter(names = {"--cache-name"})
        private String cacheName = "";

        @Parameter(names = {"--sample-min-part-size"})
        private double sampleMinPartSize = 0.1;

        @Parameter(names = {"--sample-max-part-size"})
        private double sampleMaxPartSize = 1.0;

        @Parameter(names = {"--sample-part-size-step"})
        private double samplePartSizeStep = 0.1;

        @Parameter(names = {"--trainers"}, description = "list of values from set [rf, svm, dt, bst]")
        private List<String> trainers = Collections.singletonList("rf");

        @Parameter(names = {"--ip-pool"})
        private List<String> ipPool = Collections.singletonList("127.0.0.1:47500..47509");

        @Parameter(names = {"--seed"})
        private Long seed = System.currentTimeMillis();

        @Parameter(names = {"--out"})
        private String outFileName = "timings.csv";
    }

    private static Ignite igniteInstance;

    public static void main(String[] argsList) {
        Args args = new Args();
        JCommander.newBuilder()
            .addObject(args).build()
            .parse(argsList);
        normalizeArgs(args);

        try (Ignite ignite = Ignition.start(fillConfig(args))) {
            igniteInstance = ignite;
            tryDestroyCache(args, ignite);
            IgniteCache<Integer, VectorWithAswer> trainset = fillCache(args, ignite);
            startBenckmark(args, trainset);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private static void normalizeArgs(Args args) {
        args.sampleMinPartSize = Math.max(0.0, Math.min(1.0, args.sampleMinPartSize));
        args.sampleMaxPartSize = Math.min(1.0, Math.max(0.0, args.sampleMaxPartSize));
        args.samplePartSizeStep = Math.max(0.01, Math.min(0.99, args.samplePartSizeStep));

        if (args.trainers.isEmpty())
            throw new IllegalArgumentException("Traniners list cannot be empty");
        args.trainers = args.trainers.stream().filter(x -> algorithms.contains(x)).collect(Collectors.toList());
    }

    private static IgniteConfiguration fillConfig(Args args) {
        IgniteConfiguration config = new IgniteConfiguration();
        TcpDiscoverySpi spi = new TcpDiscoverySpi();

        TcpDiscoveryVmIpFinder finder = new TcpDiscoveryVmIpFinder();
        finder.setAddresses(args.ipPool);

        spi.setIpFinder(finder);
        config.setDiscoverySpi(spi);
        config.setClientMode(false);

        return config;
    }

    private static void tryDestroyCache(Args args, Ignite ignite) {
        try {
            ignite.destroyCache(args.cacheName);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static IgniteCache<Integer, VectorWithAswer> fillCache(Args args, Ignite ignite) throws IOException {
        System.out.println(">>>>>>>>>>>>>>>>>>>>>> CREATE AND FILL DATA_CACHE");

        CacheConfiguration<Integer, VectorWithAswer> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));
        cacheConfiguration.setName(args.cacheName);
        IgniteCache<Integer, VectorWithAswer> cache = ignite.createCache(cacheConfiguration);

        AtomicInteger counter = new AtomicInteger(0);
        Files.lines(Paths.get(args.samplePath)).skip(1).forEach(line -> {
            cache.put(counter.getAndIncrement(), new VectorWithAswer(line));
        });

        return cache;
    }

    private static void startBenckmark(Args args, IgniteCache<Integer, VectorWithAswer> trainset) {
        System.out.println(">>>>>>>>>>>>>>>>>>>>>> START BENCHMARK");
        printHeader(args);
        for(String trainerName : args.trainers) {
            DatasetTrainer<? extends Model<Vector, Double>, Double> trainer = createTrainer(trainerName);
            for(double partSize = args.sampleMinPartSize;
                partSize <= args.sampleMaxPartSize;
                partSize += args.samplePartSizeStep) {

                long delta = 0L;
                for(int i = 0; i < COUNT_OF_ESTIMATIONS_PER_CASE; i++)
                    delta += estimateTs(trainer, trainset, partSize, args);
                delta /= COUNT_OF_ESTIMATIONS_PER_CASE;

                printEstimationResult(partSize, args, trainerName, delta);
            }

        }

        System.out.println(">>>>>>>>>>>>>>>>>>>>>> DONE");
    }

    private static DatasetTrainer<? extends Model<Vector, Double>, Double> createTrainer(String name) {
        switch (name) {
            case "rf":
                return new RandomForestClassifierTrainer(
                    900, 5,
                    1000, 0.1, 3,
                    0.0
                );
            case "dt":
                return new DecisionTreeClassificationTrainer(10, 0.0);
            case "bst":
                return new GDBBinaryClassifierOnTreesTrainer(1.0, 1000, 3, 0.0);
            case "svm":
                return new SVMLinearBinaryClassificationTrainer();
            default:
                throw new IllegalArgumentException(name);
        }
    }

    private static long estimateTs(DatasetTrainer<? extends Model<Vector, Double>, Double> trainer,
        IgniteCache<Integer, VectorWithAswer> trainset,
        double size, Args args) {

        String trainerName = trainer.getClass().getSimpleName();
        System.out.println(String.format(">>>>>>>>>>>>>>>>>>>>>> START LEARNING of %s [part size = %.4f]", trainerName, size));
        Random rnd = new Random(args.seed);
        SHA256UniformMapper<Integer, VectorWithAswer> sampleFilter = new SHA256UniformMapper<>(rnd);
        long startTime = System.currentTimeMillis();
        trainer.fit(
            igniteInstance,
            trainset,
            (k,v) -> sampleFilter.map(k,v) < size,
            (k,v) -> v.features(),
            (k,v) -> v.answer()
        );

        return System.currentTimeMillis() - startTime;
    }

    private static void printHeader(Args args) {
        try(FileOutputStream fos = new FileOutputStream(args.outFileName, false);
            PrintWriter printer = new PrintWriter(fos)) {

            printer.println("trainer\ttime_delta_in_ms\tpart_size");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printEstimationResult(double size, Args args, String trainerName, long timeDelta) {
        try(FileOutputStream fos = new FileOutputStream(args.outFileName, true);
            PrintWriter printer = new PrintWriter(fos)) {

            printer.println(String.format("%s\t%d\t%.4f", trainerName, timeDelta, size));
        } catch (IOException e) {
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
