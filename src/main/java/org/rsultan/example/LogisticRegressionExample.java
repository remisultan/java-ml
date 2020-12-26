package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.LogisticRegression;

import java.io.IOException;

public class LogisticRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var df = Dataframes.csv(args[0], ",", false);
        var testDf = Dataframes.csv(args[1], ",", false);

        var softmaxRegression = new LogisticRegression(1000, 0.1)
                .setResponseVariableName("c4")
                .setPredictorNames(
                        "c0", "c1", "c2", "c3"
                ).train(df);
        softmaxRegression.getHistory().tail();
        softmaxRegression.predict(testDf).show(2000);
    }
}
