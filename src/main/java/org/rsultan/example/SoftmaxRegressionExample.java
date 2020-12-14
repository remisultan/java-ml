package org.rsultan.example;

import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.SoftmaxRegression;

import java.util.List;
import java.util.stream.Collectors;

public class SoftmaxRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) {
        var categories = List.of("1", "1", "2", "2", "3", "1", "2", "3", "1");
        var df = Dataframes
                .create(
                        new Column<>("categories", categories),
                        new Column<>("x1", categories.stream().map(cat -> RandomUtils.nextDouble(0, 10) ).collect(Collectors.toList())),
                        new Column<>("x2", categories.stream().map(cat -> RandomUtils.nextDouble(0, 10)).collect(Collectors.toList())),
                        new Column<>("x3", categories.stream().map(cat -> RandomUtils.nextDouble(0, 10)).collect(Collectors.toList()))
                );
        df.show(100);

        var softmax = new SoftmaxRegression(400, 1E-3)
                .setResponseVariableName("categories")
                .setPredictorNames("x1", "x2", "x3")
                .train(df);
    }
}
