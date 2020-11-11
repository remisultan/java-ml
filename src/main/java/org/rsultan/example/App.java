package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframe.Column;
import org.rsultan.regression.LinearRegression;

import java.util.List;
import java.util.UUID;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

/**
 * Hello world!
 */
public class App {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) {
        List<Integer> integers = range(0, 1000).boxed().collect(toList());
        Dataframe.create(
                new Column("Integers", integers),
                new Column("UUIDs", range(0, 1000).boxed().map(num -> UUID.randomUUID()).collect(toList()))
        )
                .withColumn("Integers", "Exponential", (Function<Integer, Double>) Math::exp)
                .withColumn("Integers", "Exponential", "Exponential * i",
                        (Integer i, Double exponential) -> exponential * i.doubleValue())
                .show(20);


        // int size = 1000;
        // INDArray O = Nd4j.ones(size, 1);
        // double[] data = range(0, size).asDoubleStream().map(num -> Math.pow(num, 2) + nextDouble(0, 100) * (nextBoolean() ? 1 : -1)).toArray();
        // var Y = Nd4j.create(data, size, 1);
        // double[] xValues = range(0, size).asDoubleStream().toArray();
        // double[] x2Values = range(0, size).asDoubleStream()
        //         .map(num -> Math.pow(num, 2))
        //         .toArray();
        // var X = Nd4j.concat(1, O,
        //         Nd4j.create(xValues, size, 1),
        //         Nd4j.create(x2Values, size, 1)
        // );
        // var regression = new LinearRegression().train(X, Y);
//
        // System.out.println("BETA: " + regression.getBETA());
        // System.out.println("R2: " + regression.getR2());
        // System.out.println("MSE: " + regression.getMSE());
        // System.out.println("RMSE: " + regression.getRMSE());
        // System.out.println("Tvalues : " + regression.gettValues());
        // System.out.println("Pvalues : " + regression.getpValues());
    }

    private static void testRandom() {
        int size = 10;
        INDArray O = Nd4j.ones(size, 1);

        var Y = Nd4j.create(range(0, size).asDoubleStream().toArray(), new int[]{size, 1});
        var X = Nd4j.concat(1, O, Nd4j.rand(size, 3));
        var regression = new LinearRegression().train(X, Y);
        System.out.println("BETA: " + regression.getBETA());
        System.out.println("R2: " + regression.getR2());
        System.out.println("MSE: " + regression.getMSE());
        System.out.println("RMSE: " + regression.getRMSE());
        System.out.println("Tvalues : " + regression.gettValues());
        System.out.println("Pvalues : " + regression.getpValues());
    }
}
