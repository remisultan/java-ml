package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;

public class IsolationForestExample {

  /*
    You can use the http dataset --> args[0]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", true);

    IsolationForest model = new IsolationForest(200, 0.8).train(df.mapWithout("attack"));
    var anomalies = df.filter("attack", (Long i) -> i == 1L).mapWithout("attack");
    var nonAnomalies = Dataframes.trainTest(df.getColumns())
        .setSplitValue(0.99).split()
        .test().filter("attack", (Long i) -> i == 0L).mapWithout("attack");

    model.predict(anomalies).tail();
    model.predict(nonAnomalies).tail();
  }
}
