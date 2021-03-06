package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.core.dimred.PrincipalComponentAnalysis;
import org.rsultan.dataframe.Dataframes;

public class PCAExample {

  /*
    You can use the iris dataset --> args[0]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", false);
    var PCA = new PrincipalComponentAnalysis(1)
        .setResponseVariable("c4")
        .train(df);

    var predictions = PCA.predict(df);
    var reconstruct = PCA.reconstruct();

    predictions.show(0, 10);
    reconstruct.show(0, 10);

    System.out.println(Transforms
        .cosineSim(
            df.mapWithout("c4").toMatrix(),
            reconstruct.mapWithout("c4").toMatrix()
        )
    );
  }
}
