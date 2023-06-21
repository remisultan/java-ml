package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.core.dimred.PrincipalComponentAnalysis;
import org.rsultan.dataframe.Dataframes;

public class PCATest {

  /*
    You can use the iris dataset --> args[0]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", true);
    var PCA = new PrincipalComponentAnalysis(2)
        .setResponseVariable("malicious_ip")
        .train(df);

    var predictions = PCA.predict(df);

    predictions.write("target/pca_features.csv", ",", "");
  }
}
