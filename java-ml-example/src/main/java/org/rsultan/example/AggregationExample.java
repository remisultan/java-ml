package org.rsultan.example;

import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.engine.mapper.impl.group.Aggregation;
import org.rsultan.dataframe.engine.mapper.impl.group.AggregationType;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class AggregationExample {

  /*
   The data used is the infamous IRIS dataset
   Make sure args[0] & args[1] /path/to/your/src/main/resources/softmax/iris.data
  */
  public static void main(String[] args) {
    var df = Dataframes.csv(args[0], ",", "\"", false);

    df.groupBy("c4",
            new Aggregation("c1", AggregationType.SUM),
            new Aggregation("c1", AggregationType.AVG),
            new Aggregation("c1", AggregationType.COUNT),
            new Aggregation("c1", AggregationType.MAX),
            new Aggregation("c1", AggregationType.MIN),
            new Aggregation("c1", AggregationType.ACCUMULATE)

        )
        .show(100);
  }
}
