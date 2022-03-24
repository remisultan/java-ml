package org.rsultan.example;

import static org.rsultan.core.regularization.Regularization.LASSO;
import static org.rsultan.core.regularization.Regularization.RIDGE;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.regression.impl.LogisticRegression;
import org.rsultan.dataframe.Dataframes;

public class LogisticRegressionExample {

  /*
   The data used is the infamous IRIS dataset
   Make sure args[0] & args[1] /path/to/your/src/main/resources/softmax/iris.data
  */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", false);

    var setosaRegression = new LogisticRegression(1000, 0.1)
        .setResponseVariableName("c4")
        .setPredictorNames("c0", "c1", "c2", "c3")
        .setLossAccuracyOffset(100)
        .setRegularization(RIDGE)
        .setLambda(0.019)
        .setChosenLabel("Iris-setosa")
        .train(df);
    setosaRegression.getHistory().show(1000);
    setosaRegression.predict(df).show(1000);

    var versicolorRegression = new LogisticRegression(1000, 0.1)
        .setResponseVariableName("c4")
        .setPredictorNames("c0", "c1", "c2", "c3")
        .setChosenLabel("Iris-versicolor")
        .setLossAccuracyOffset(100)
        .train(df);
    setosaRegression.getHistory().show(1000);
    versicolorRegression.predict(df).show(1000);

    var virginicaRegression = new LogisticRegression(1000, 0.1)
        .setResponseVariableName("c4")
        .setPredictorNames("c0", "c1", "c2", "c3")
        .setChosenLabel("Iris-virginica")
        .setRegularization(LASSO)
        .setLambda(0.0015)
        .setLossAccuracyOffset(100)
        .train(df);
    virginicaRegression.getHistory().show(1000);
    virginicaRegression.predict(df).show(1000);
  }
}
