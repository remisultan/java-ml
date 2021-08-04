package org.rsultan.example;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.dbscan.DBSCAN;
import org.rsultan.dataframe.Dataframes;

public class DBSCANExample {

  /*
    You can use the iris dataset --> args[0]
   */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var df = Dataframes.csv(args[0], ",", "\"", false);

    var predicted = new DBSCAN(1, 5)
        .predict(df.mapWithout("c4"));

    predicted.addColumn(df.getColumns()[4]).show(150);
  }
}
