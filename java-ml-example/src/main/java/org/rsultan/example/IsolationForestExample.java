package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.evaluation.AreaUnderCurve;
import org.rsultan.core.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;

public class IsolationForestExample {

  /*
    You can use the http dataset --> args[0]
    You can use the http_reduced.csv dataset for testing --> args[1]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", true);
    var testDf = Dataframes.csv(args[1], ",", "\"", true);

    var model = new IsolationForest(200).setUseAnomalyScoresOnly(true);
    var evaluator = new AreaUnderCurve<IsolationForest>()
        .setTrainTestThreshold(0.75)
        .setTestDataframe(testDf)
        .setLearningRate(0.01);
    evaluator.evaluate(model, df);
    System.out.println(evaluator.getAUC());
  }
}
