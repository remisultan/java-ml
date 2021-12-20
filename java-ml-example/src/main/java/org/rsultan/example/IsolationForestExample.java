package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

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
    var trainTestDataframe = Dataframes.trainTest(df.getColumns()).setSplitValue(0.99);

    var model = new IsolationForest(200);
    var evaluator = new TPRThresholdEvaluator("attack", "anomalies")
            .setDesiredTPR(0.7)
            .setExternalTestDataframe(testDf)
            .setLearningRate(0.1);
    Double threshold = evaluator.evaluate(model, trainTestDataframe);
    System.out.println("threshold = " + threshold);
    evaluator.showMetrics();
  }
}
