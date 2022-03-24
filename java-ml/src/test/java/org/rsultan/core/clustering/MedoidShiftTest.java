package org.rsultan.core.clustering;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextFloat;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.apache.commons.lang3.RandomUtils.nextLong;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelSerdeTestUtils;
import org.rsultan.core.clustering.medoidshift.MeanShift;
import org.rsultan.core.clustering.medoidshift.MedianShift;
import org.rsultan.core.clustering.medoidshift.MedoidShift;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class MedoidShiftTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_must_apply_medoid_shift() {
    return Stream.of(
        Arguments.of(new MeanShift(60D, 30)),
        Arguments.of(new MeanShift(65D, 30)),
        Arguments.of(new MeanShift(70D, 30)),
        Arguments.of(new MedianShift(60D, 30)),
        Arguments.of(new MedianShift(65D, 30)),
        Arguments.of(new MedianShift(70D, 30))
    );
  }

  @ParameterizedTest
  @MethodSource("params_must_apply_medoid_shift")
  public void must_apply_medoid_shift(MedoidShift medoidShift) {
    var dataframe = Dataframes.create(
        new String[]{"c1", "c2", "c3", "c4"}, range(0, 100).mapToObj(idx ->
            List.of(nextLong(0, 100), nextDouble(0, 100), nextFloat(0, 100), nextInt(0, 100))
        ).collect(toList()));
    medoidShift = medoidShift.train(dataframe);
    medoidShift.showMetrics();

    assertThat(medoidShift.getCentroids()).isNotNull();
    assertThat(medoidShift.predict(dataframe)).isNotNull();
  }

  @ParameterizedTest
  @MethodSource("params_must_apply_medoid_shift")
  public void must_serde_apply_medoid_shift(MedoidShift medoidShift) {
    var dataframe = Dataframes.create(
        new String[]{"c1", "c2", "c3", "c4"}, range(0, 100).mapToObj(idx ->
            List.of(nextLong(0, 100), nextDouble(0, 100), nextFloat(0, 100), nextInt(0, 100))
        ).collect(toList()));
    medoidShift = ModelSerdeTestUtils.serdeTrainable(medoidShift.train(dataframe));
    medoidShift.showMetrics();

    assertThat(medoidShift.getCentroids()).isNotNull();
    assertThat(medoidShift.predict(dataframe)).isNotNull();
  }
}
