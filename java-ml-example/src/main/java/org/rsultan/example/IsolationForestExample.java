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

    new IsolationForest(200, 0.6)
        .train(df.mapWithout("attack"))
        .predict(df.filter("attack", (Long i) -> i == 1L).mapWithout("attack"))
        .show(1000000);
  }
}
