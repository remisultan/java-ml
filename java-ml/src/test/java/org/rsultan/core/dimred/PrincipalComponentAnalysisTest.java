package org.rsultan.core.dimred;

import static org.assertj.core.api.Assertions.assertThat;
import static org.nd4j.linalg.ops.transforms.Transforms.cosineSim;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelSerdeTestUtils;
import org.rsultan.dataframe.Dataframe.Result;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;

public class PrincipalComponentAnalysisTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_must_apply_PCA_to_dataframe() {
    return Stream.of(
        Arguments.of(new PrincipalComponentAnalysis(1).setResponseVariable("strColumn"), 2,
            0.9997709698637104),
        Arguments.of(new PrincipalComponentAnalysis(2).setResponseVariable("strColumn"), 3,
            0.9999991213055501),
        Arguments.of(new PrincipalComponentAnalysis(3).setResponseVariable("strColumn"), 4,
            0.9999991213055501),
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

    var result = predictions.getResult();
    assertThat(result.header().size()).isEqualTo(expectedColumns);
    assertThat(result.rows().size()).isEqualTo(5);

    var result1 = reconstruct.getResult();
    assertThat(result1.header().size()).isEqualTo(5);
    assertThat(result1.rows().size()).isEqualTo(result.rows().size());

    assertThat(cosineSim(reconstruct.mapWithout("strColumn").toMatrix(),
        df.mapWithout("strColumn").toMatrix())).isLessThanOrEqualTo(reconstructSimilarity);
  }

  @ParameterizedTest
  @MethodSource("params_must_apply_PCA_to_dataframe")
  public void must_serde_apply_PCA_to_dataframe(
      PrincipalComponentAnalysis pca,
      int expectedColumns,
      double reconstructSimilarity
  ) throws IOException {
    var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-pca.csv"));
    pca = ModelSerdeTestUtils.serdeTrainable(pca.train(df));
    var predictions = pca.predict(df);
    var reconstruct = pca.reconstruct();

    var result = predictions.getResult();
    assertThat(result.header().size()).isEqualTo(expectedColumns);
    assertThat(result.rows().size()).isEqualTo(5);

    var result1 = reconstruct.getResult();
    assertThat(result1.header().size()).isEqualTo(5);
    assertThat(result1.rows().size()).isEqualTo(result.rows().size());

    assertThat(cosineSim(reconstruct.mapWithout("strColumn").toMatrix(),
        df.mapWithout("strColumn").toMatrix())).isLessThanOrEqualTo(reconstructSimilarity);
  }
}
