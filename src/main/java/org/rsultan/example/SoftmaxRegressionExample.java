package org.rsultan.example;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.impl.SoftmaxRegression;
import org.rsultan.regularization.Regularization;

import java.io.IOException;

import static org.rsultan.regularization.Regularization.LASSO;
import static org.rsultan.regularization.Regularization.RIDGE;

public class SoftmaxRegressionExample {

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        var df = Dataframes.csv(args[0], ",", false);
        var testDf = Dataframes.csv(args[1], ",", false);

        var softmaxRegression = new SoftmaxRegression(1000, 0.1)
                .setResponseVariableName("c4")
                .setPredictorNames("c0", "c1", "c2", "c3")
                .setRegularization(RIDGE)
                .setLambda(0.0014)
                .setLossAccuracyOffset(100)
                .train(df);
        softmaxRegression.getHistory().tail();
        softmaxRegression.predict(testDf).show(2000);

        softmaxRegression = new SoftmaxRegression(1000, 0.1)
                .setResponseVariableName("c4")
                .setPredictorNames("c0", "c1", "c2", "c3")
                .setRegularization(LASSO)
                .setLambda(0.0014)
                .setLossAccuracyOffset(100)
                .train(df);
        softmaxRegression.getHistory().tail();
        softmaxRegression.predict(testDf).show(2000);
    }
}
