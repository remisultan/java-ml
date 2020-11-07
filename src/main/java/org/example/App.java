package org.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.regression.LinearRegression;
import java.util.stream.IntStream;

import static java.util.stream.IntStream.range;
import static org.apache.commons.lang3.ArrayUtils.toPrimitive;

/**
 * Hello world!
 */
public class App {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }
    public static void main(String[] args) {
        int size = 10;
        INDArray O = Nd4j.ones(size, 1);
        double[] data =  toPrimitive(range(0, size).asDoubleStream().boxed().map(num -> Math.pow(num, 5)).toArray(Double[]::new));
        var Y = Nd4j.create(data, size, 1);
        double[] xValues = range(0, size).asDoubleStream().toArray();
        double[] x2Values = toPrimitive(range(0, size).asDoubleStream().boxed().map(num -> Math.pow(num, 4)).toArray(Double[]::new));
        var X = Nd4j.concat(1, O,
                Nd4j.create(xValues, size, 1)
             //   Nd4j.create(x2Values, size, 1)
        );
        var regression = new LinearRegression().train(X,Y);

        System.out.println("BETA: " + regression.getBETA());
        System.out.println("R2: " + regression.getR2());
        System.out.println("Tvalues : " + regression.gettValues());
        System.out.println("Pvalues : " + regression.getpValues());
    }

    private static void testRandom() {
        int size = 10;
        INDArray O = Nd4j.ones(size, 1);

        var Y = Nd4j.create(range(0, size).asDoubleStream().toArray(), new int[]{size, 1});
        var X = Nd4j.concat(1, O, Nd4j.rand(size, 2));
        var regression = new LinearRegression().train(X,Y);
        System.out.println("BETA: " + regression.getBETA());
        System.out.println("R2: " + regression.getR2());
        System.out.println("Tvalues : " + regression.gettValues());
        System.out.println("Pvalues : " + regression.getpValues());
    }
}
