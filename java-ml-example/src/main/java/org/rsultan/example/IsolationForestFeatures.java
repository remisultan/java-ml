package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;

public class IsolationForestFeatures {

  /*
    You can use the http dataset --> args[0]
    You can use the http_reduced.csv dataset for testing --> args[1]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", true);

    var model = new IsolationForest(200)
        .setSampleSize(4096)
        .setUseAnomalyScoresOnly(true);

    model.train(df.copy().mapWithout("malicious_ip").shuffle())
        .predict(df.copy().mapWithout("malicious_ip"))
        .transform("anomalies", (Double score) ->  1.0D - score)
        .addColumn("malicious_ip", df.copy().select("malicious_ip").getResult().rows().stream().map(row -> row.values().get(0)).toList())
        .write("target/anomaly_scores.csv", ",", "");
  }
}
