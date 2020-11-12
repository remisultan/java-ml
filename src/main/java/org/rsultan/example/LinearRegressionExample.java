package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.regression.LinearRegression;

import java.io.IOException;

/**
 * Hello world!
 */
public class LinearRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var dataframe = Dataframe.csv(args[0], ",", true);

        dataframe.show(20);
        new LinearRegression()
                .setResponseVariableName("fare_amount")
                .setPredictorNames("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count")
                .train(dataframe)
                .showMetrics();
    }
}
