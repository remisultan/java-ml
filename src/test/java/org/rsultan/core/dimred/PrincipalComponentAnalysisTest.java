package org.rsultan.core.dimred;

import static org.assertj.core.api.Assertions.assertThat;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframes;

public class PrincipalComponentAnalysisTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_must_apply_PCA_to_dataframe() {
    return Stream.of(
        Arguments.of(new PrincipalComponentAnalysis(1).setResponseVariable("strColumn"), 2, 0.708),
        Arguments.of(new PrincipalComponentAnalysis(2).setResponseVariable("strColumn"), 3, 0.708),
        Arguments.of(new PrincipalComponentAnalysis(3).setResponseVariable("strColumn"), 4, 0.709),
        Arguments.of(new PrincipalComponentAnalysis(4).setResponseVariable("strColumn"), 5, 1.0)
    );
  }

  @ParameterizedTest
  @MethodSource("params_must_apply_PCA_to_dataframe")
  public void must_apply_PCA_to_dataframe(
      PrincipalComponentAnalysis pca,
      int expectedColumns,
      double reconstructSimilarity
  ) throws IOException {
    var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-pca.csv"));
    pca.train(df);
    var predictions = pca.predict(df);
    var reconstruct = pca.reconstruct();

    assertThat(predictions.getColumnSize()).isEqualTo(expectedColumns);
    assertThat(predictions.getRowSize()).isEqualTo(df.getRowSize());

    assertThat(reconstruct.getColumnSize()).isEqualTo(df.getColumnSize());
    assertThat(reconstruct.getRowSize()).isEqualTo(df.getRowSize());

    assertThat(cosineSim(reconstruct.mapWithout("strColumn").toMatrix(),
        df.mapWithout("strColumn").toMatrix())).isLessThanOrEqualTo(reconstructSimilarity);
  }
}
