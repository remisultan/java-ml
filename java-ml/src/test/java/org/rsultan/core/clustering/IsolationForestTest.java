package org.rsultan.core.clustering;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.rangeClosed;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;

public class IsolationForestTest {

  private static Row[] rows;

  static {
    rows = List.of(0, 3, 6, 7, 8).stream()
        .map(IsolationForestTest::createCircleDataAroundCenter)
        .flatMap(List::stream)
        .toArray(Row[]::new);
  }

  private static List<Row> createCircleDataAroundCenter(int radius) {
    return rangeClosed(-radius, radius).boxed().flatMap(x -> {
      double y = Math.sqrt(radius * radius - x * x);
      return List.of(new Row(x, y), new Row(x, -y)).stream();
    }).collect(toList());
  }

  @Test
  public void must_perform_isolation() {
    var df = Dataframes.create(new String[]{"x", "y"}, rows);
    var predict = new IsolationForest(10, 0.56).setSampleSize(15).train(df).predict(df);
    int anomalies = predict.filter("anomalies", TRUE::equals).getRowSize();
    int nonAnomalies = predict.filter("anomalies", FALSE::equals).getRowSize();

    assertThat(anomalies).isBetween(1, predict.getRowSize());
    assertThat(nonAnomalies).isEqualTo(predict.getRowSize() - anomalies);
  }


}
