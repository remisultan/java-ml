package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
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
public class LinearRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) {
        int size = 1000;

        var yList = range(0, size)
                .asDoubleStream()
                .map(num -> Math.pow(num, 6) + nextDouble(0, 100) * (nextBoolean() ? 1 : -1))
                .boxed().collect(toList());

        var xList = range(0, size).asDoubleStream().boxed().collect(toList());

        var dataframe = Dataframe
                .create(new Column("Y", yList), new Column("x", xList))
                .withColumn("x^2", "x", (Double x) -> x * x)
                .withColumn("x^3", "x^2", (Double x) -> x * x);

        dataframe.show(size);

        new LinearRegression()
                .setResponseVariableName("Y")
                .setPredictorNames("x", "x^2", "x^3")
                .train(dataframe)
                .showMetrics();

    }
}
