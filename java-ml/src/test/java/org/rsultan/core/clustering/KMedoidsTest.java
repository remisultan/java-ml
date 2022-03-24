package org.rsultan.core.clustering;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextFloat;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.apache.commons.lang3.RandomUtils.nextLong;
import static org.assertj.core.api.Assertions.assertThat;
import static org.rsultan.core.ModelSerdeTestUtils.serdeTrainable;

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.kmedoids.KMedoids;
import org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class KMedoidsTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_apply_kmedoids() {
    return Stream.of(
        Arguments.of(new KMeans(3, 10)),
        Arguments.of(new KMeans(3, 10, InitialisationStrategy.RANDOM)),
        Arguments.of(new KMeans(1, 10)),
        Arguments.of(new KMeans(1, 10, InitialisationStrategy.RANDOM)),
        Arguments.of(new KMeans(2, 10)),
        Arguments.of(new KMeans(2, 10, InitialisationStrategy.RANDOM)),
        Arguments.of(new KMedians(3, 10)),
        Arguments.of(new KMedians(3, 10, InitialisationStrategy.RANDOM)),
        Arguments.of(new KMedians(1, 10)),
        Arguments.of(new KMedians(1, 10, InitialisationStrategy.RANDOM)),
        Arguments.of(new KMedians(2, 10)),
        Arguments.of(new KMedians(2, 10, InitialisationStrategy.RANDOM))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_kmedoids")
  public void must_apply_kmedoids(KMedoids kMedoids) {
    var dataframe = Dataframes.create(
        new String[]{"c1", "c2", "c3", "c4"}, range(0, 100).mapToObj(idx ->
            List.of(nextLong(0, 100), nextDouble(0, 100), nextFloat(0, 100), nextInt(0, 100))
        ).collect(toList()));
    kMedoids.train(dataframe);
    kMedoids.showMetrics();

    assertThat(kMedoids.getK()).isNotNull();
    assertThat(kMedoids.getCentroids()).isNotNull();
    assertThat(kMedoids.getCluster()).isNotNull();
    assertThat(kMedoids.getCentroids().rows()).isEqualTo(kMedoids.getK());
    assertThat(kMedoids.getLoss()).isNotNull();
    assertThat(kMedoids.getWCSS()).isNotNull();
    assertThat(kMedoids.predict(dataframe)).isNotNull();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_kmedoids")
  public void must_serde_and_apply_kmedoids(KMedoids kMedoids) {
    var dataframe = Dataframes.create(
        new String[]{"c1", "c2", "c3", "c4"}, range(0, 100).mapToObj(idx ->
            List.of(nextLong(0, 100), nextDouble(0, 100), nextFloat(0, 100), nextInt(0, 100))
        ).collect(toList()));
    var kMedoid = serdeTrainable(kMedoids.train(dataframe));
    kMedoid.showMetrics();

    assertThat(kMedoid.getK()).isNotNull();
    assertThat(kMedoid.getCentroids()).isNotNull();
    assertThat(kMedoid.getCluster()).isNotNull();
    assertThat(kMedoid.getCentroids().rows()).isEqualTo(kMedoids.getK());
    assertThat(kMedoid.getLoss()).isNotNull();
    assertThat(kMedoid.getWCSS()).isNotNull();
    assertThat(kMedoid.predict(dataframe)).isNotNull();
  }
}
