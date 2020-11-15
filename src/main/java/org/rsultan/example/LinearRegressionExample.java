package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.LinearRegression;

import java.io.IOException;

public class LinearRegressionExample {

    /*
        The data used is a sample of https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data
        Make sure args[0] /path/to/your/src/main/resources/linear/train.csv and
        Also that args[1] /path/to/your/src/main/resources/linear/test.csv
     */
    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var df = Dataframes.csv(args[0]);
        var testDf = Dataframes.csv(args[1]);
        df.show(20);
        testDf.show(20);

        var linearRegression = new LinearRegression()
                .setResponseVariableName("fare_amount")
                .setPredictorNames("pickup_longitude",
                        "pickup_latitude",
                        "dropoff_longitude",
                        "dropoff_latitude",
                        "passenger_count")
                .train(df)
                .showMetrics();

        linearRegression.predict(testDf).show(20);
    }
}
