package org.rsultan.core.clustering;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextFloat;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.apache.commons.lang3.RandomUtils.nextLong;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.kmedoids.KMedoids;
import org.rsultan.core.clustering.medoidshift.MeanShift;
import org.rsultan.core.clustering.medoidshift.MedianShift;
import org.rsultan.core.clustering.medoidshift.MedoidShift;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class MedoidShiftTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_apply_kmedoids() {
    return Stream.of(
        Arguments.of(new MeanShift(60, 30)),
        Arguments.of(new MeanShift(65, 30)),
        Arguments.of(new MeanShift(70, 30)),
        Arguments.of(new MedianShift(60, 30)),
        Arguments.of(new MedianShift(65, 30)),
        Arguments.of(new MedianShift(70, 30))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_kmedoids")
  public void must_apply_kmedoids(MedoidShift medoidShift) {
    var dataframe = Dataframes.create(
        new Column<>("c1", range(0, 100).map(idx -> nextLong(0, 100)).boxed().collect(toList())),
        new Column<>("c2",
            range(0, 100).mapToDouble(idx -> nextDouble(0, 100)).boxed().collect(toList())),
        new Column<>("c3", range(0, 100).boxed().map(idx -> nextFloat(0, 100)).collect(toList())),
        new Column<>("c4", range(0, 100).boxed().map(idx -> nextInt(0, 100)).collect(toList()))
    );
    medoidShift.train(dataframe);

    assertThat(medoidShift.getC()).isNotNull();
    assertThat(medoidShift.predict(dataframe)).isNotNull();
  }
}