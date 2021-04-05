package org.rsultan.core.clustering;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextFloat;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.apache.commons.lang3.RandomUtils.nextLong;
import static org.assertj.core.api.Assertions.assertThat;
import static org.rsultan.core.clustering.type.MedoidType.MEAN;
import static org.rsultan.core.clustering.type.MedoidType.MEDIAN;

import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.evaluation.KMedoidEvaluator;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class KMedoidsEvaluatorTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_apply_kmedoids() {
    return Stream.of(
        Arguments.of(new KMedoidEvaluator(2, 3, MEAN)),
        Arguments.of(new KMedoidEvaluator(2, 10, MEAN)),
        Arguments.of(new KMedoidEvaluator(2, 3, MEDIAN)),
        Arguments.of(new KMedoidEvaluator(2, 10, MEDIAN)),
        Arguments.of(new KMedoidEvaluator(2, 3, MEDIAN, 10)),
        Arguments.of(new KMedoidEvaluator(2, 10, MEDIAN, 10)),
        Arguments.of(new KMedoidEvaluator(2, 3, MEAN, 10)),
        Arguments.of(new KMedoidEvaluator(2, 10, MEAN, 10))
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_apply_kmedoids")
  public void must_apply_kmedoids(KMedoidEvaluator kMedoidEvaluator) {
    var dataframe = Dataframes.create(
        new Column<>("c1", range(0, 1000).map(idx -> nextLong(0, 1000)).boxed().collect(toList())),
        new Column<>("c2",
            range(0, 1000).mapToDouble(idx -> nextDouble(0, 1000)).boxed().collect(toList())),
        new Column<>("c3", range(0, 1000).boxed().map(idx -> nextFloat(0, 1000)).collect(toList())),
        new Column<>("c4", range(0, 1000).boxed().map(idx -> nextInt(0, 1000)).collect(toList()))
    );
    var targetDataframe = kMedoidEvaluator.evaluate(dataframe);
    targetDataframe.tail();

    assertThat(targetDataframe.getData()).hasSize(2);
    assertThat(targetDataframe.getData().get("K")).isNotEmpty();
    assertThat(targetDataframe.getData().get("sumOfSquares")).isNotEmpty();
  }
}
