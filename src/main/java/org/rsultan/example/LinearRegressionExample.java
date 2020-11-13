package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.LinearRegression;

import java.io.IOException;

public class LinearRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var dataframe = Dataframes.csv(args[0]);
        dataframe.show(20);
        new LinearRegression()
                .setResponseVariableName("fare_amount")
                .setPredictorNames("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count")
                .train(dataframe)
                .showMetrics();
    }
}
