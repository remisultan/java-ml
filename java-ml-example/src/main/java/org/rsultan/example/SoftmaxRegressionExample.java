package org.rsultan.example;

import static org.rsultan.core.regularization.Regularization.LASSO;
import static org.rsultan.core.regularization.Regularization.RIDGE;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.regression.impl.SoftmaxRegression;
import org.rsultan.dataframe.Dataframes;

public class SoftmaxRegressionExample {


  /*
   The data used is the infamous IRIS dataset
   Make sure args[0] & args[1] /path/to/your/src/main/resources/softmax/iris.data
  */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", false);

    var softmaxRegression = new SoftmaxRegression(1000, 0.1)
        .setResponseVariableName("c4")
        .setPredictorNames("c0", "c1", "c2", "c3")
        .setRegularization(RIDGE)
        .setLambda(0.005)
        .setLossAccuracyOffset(100)
        .train(df);
    softmaxRegression.getHistory().show(1000);
    softmaxRegression.predict(df.copy()).show(2000);

    softmaxRegression = new SoftmaxRegression(1000, 0.1)
        .setResponseVariableName("c4")
        .setPredictorNames("c0", "c1", "c2", "c3")
        .setRegularization(LASSO)
        .setLambda(0.003)
        .setLossAccuracyOffset(100)
        .train(df);
    softmaxRegression.getHistory().show(1000);
    softmaxRegression.predict(df.copy()).show(2000);
  }
}
