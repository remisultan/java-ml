package org.rsultan.core.clustering;


import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.rangeClosed;
import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.rsultan.core.clustering.dbscan.DBSCAN;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;

public class DBSCANTest {

  private static Row[] rows;

  static {
    rows = List.of(0, 3, 6, 7, 8).stream()
        .map(DBSCANTest::createCircleDataAroundCenter)
        .flatMap(List::stream)
        .toArray(Row[]::new);
  }

  private static List<Row> createCircleDataAroundCenter(int radius) {
    return rangeClosed(-radius, radius).boxed().flatMap(x -> {
      double y = Math.sqrt(radius * radius - x * x);
      return List.of(new Row(x, y), new Row(x, -y)).stream();
    }).collect(toList());
  }

  private static Stream<Arguments> params_that_must_apply_dbscan() {
    return Stream.of(
        of(-1, -3, new Integer[]{1}, new Integer[]{106}),
        of(0.6, 3, new Integer[]{1, 2, 3}, new Integer[]{94, 6}),
        of(0.5, 3, new Integer[]{1, 2, 3, 4, 5, 6, 7}, new Integer[]{1, 14, 6, 39}),
        of(0.4, 3, rangeClosed(1, 20).boxed().toArray(Integer[]::new), new Integer[]{1, 6, 39}),
        of(0.3, 3, rangeClosed(1, 32).boxed().toArray(Integer[]::new), new Integer[]{1, 6, 33}),
        of(0.2, 3, rangeClosed(1, 106).boxed().toArray(Integer[]::new), new Integer[]{1}),
        of(0.6, 2, new Integer[]{1, 2, 3}, new Integer[]{94, 6}),
        of(0.6, 3, new Integer[]{1, 2, 3}, new Integer[]{94, 6}),
        of(0.6, 4, new Integer[]{1, 2, 3}, new Integer[]{94, 6}),
        of(0.6, 5, new Integer[]{1, 2, 3, 4, 5, 6, 7}, new Integer[]{90, 1, 6})
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_dbscan")
  public void must_apply_dbscan_test(double radius, int minSample, Integer[] expectedClusters,
      Integer[] expectedDensity) {
    var dbScan = new DBSCAN(radius, minSample);
    var df = Dataframes.create(new String[]{"x", "y"}, rows);
    var predictDf = dbScan.predict(df);

    assertThat(predictDf.<Integer>get("cluster").stream().distinct().collect(toList()))
        .containsOnly(expectedClusters);
    assertThat(predictDf.<Integer>get("density").stream().distinct().collect(toList()))
        .containsOnly(expectedDensity);
  }

  @Test
  public void must_throw_NotImplementedException_when_train() {
    var dbscan = new DBSCAN(3, 4);
    var exception = assertThrows(IllegalCallerException.class,
        () -> dbscan.train(Dataframes.create()));

    assertThat(exception.getMessage())
        .isEqualTo("Directly predict since this is a clustering algorithm");
  }

}
