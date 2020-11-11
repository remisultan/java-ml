package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframe.Column;
import org.rsultan.regression.LinearRegression;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.apache.commons.lang3.RandomUtils.nextBoolean;
import static org.apache.commons.lang3.RandomUtils.nextDouble;

/**
 * Hello world!
 */
public class App {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) {
        int size = 10;

        var yList = range(0, size)
                .asDoubleStream()
                .map(num -> Math.pow(num, 2) + nextDouble(0, 1) * (nextBoolean() ? 1 : -1))
                .boxed().collect(toList());

        var xList = range(0, size).asDoubleStream().boxed().collect(toList());

        var dataframe = Dataframe
                .create(new Column("Y", yList), new Column("x", xList))
                .withColumn("Intercept", () -> 1)
                .withColumn("x^2", "x", (Double x) -> x * x);

        dataframe.show(size);

        var regression = new LinearRegression()
                .setResponseVariableName("Y")
                .setPredictorNames("Intercept","x", "x^2")
                .train(dataframe)
                .showMetrics();

    }
}
