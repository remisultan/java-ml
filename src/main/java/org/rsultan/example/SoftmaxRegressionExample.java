package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.SoftmaxRegression;

import java.io.IOException;

public class SoftmaxRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var df = Dataframes.csv(args[0]);
        var testDf = Dataframes.csv(args[1]);

        var softmaxRegression = new SoftmaxRegression(100, 0.01)
                .setResponseVariableName("quality")
                .setPredictorNames("fixed acidity",
                        "volatile acidity",
                        "citric acid",
                        "residual sugar",
                        "chlorides",
                        "free sulfur dioxide",
                        "total sulfur dioxide",
                        "density",
                        "pH",
                        "sulphates",
                        "alcohol"
                ).train(df);
        softmaxRegression.getHistory().show(10000);
    }
}
