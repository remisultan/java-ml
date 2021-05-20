package org.rsultan.dataframe;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.rsultan.dataframe.transform.split.SplitDataframe.TrainTestSplit;

public class TrainTestDataframeTest {

  private static Stream<Arguments> params_that_must_load_test_dataframe_correctly() {
    return Stream.of(
        of(new Column[]{new Column<>("Integers", 0, 1, 2, 3, 4)}, 0.75, 5, 1),
        of(new Column[]{new Column<>("Longs", 0L, 1L, 2L, 3L, 4L)}, 0.5, 5, 1),
        of(new Column[]{new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D)}, 0.2, 5, 1),
        of(new Column[]{new Column<>("Floats", 0F, 1F, 2F, 3F, 4F)}, 0.8, 5, 1),
        of(new Column[]{new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3")}, 0.75, 5, 1),
        of(new Column[]{new Column<>("Negatives", "-1.1", "-2.1", "-3", "-4.4", "-5.3")}, 0.75, 5,
            1),
        of(new Column<?>[]{
            new Column<>("Integers", 0, 1, 2, 3, 4),
            new Column<>("Longs", 0L, 1L, 2L, 3L, 4L),
            new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D),
            new Column<>("Floats", 0F, 1F, 2F, 3F, 4F),
            new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3"),
            new Column<>("Negatives", "-1.1", "-2.1", "-3", "-4.4", "-5.3")
        }, 0.75, 5, 6)
    );
  }

  private static Stream<Arguments> params_that_must_throw_IllegalArgumentException_when_split_value_is_wrong() {
    return Stream.of(of(0.0), of(1.0), of(1.1), of(-1));
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_test_dataframe_correctly")
  public void that_must_load_test_dataframe_correctly(Column<?>[] columns,
      double splitThreshold,
      int expectedRows,
      int expectedCols
  ) {
    var dataframe = Dataframes.trainTest(columns)
        .setSplitValue(splitThreshold);
    var dfSplit = dataframe.split();

    int expectedTrainSize = (int) (splitThreshold * expectedRows);

    assertTrainTest(expectedRows, expectedCols, dfSplit, expectedTrainSize);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_test_dataframe_correctly")
  public void must_load_test_dataframe_correctly_with_rows(Column<?>[] columns,
      double splitThreshold,
      int expectedRows,
      int expectedCols
  ) {
    var dataframe = Dataframes.trainTest(
        stream(columns).map(Column::columnName).toArray(String[]::new),
        range(0, columns[0].values().size()).mapToObj(idx ->
            stream(columns).map(column -> column.values().get(idx)).collect(toList())
        ).map(Row::new).toArray(Row[]::new)
    ).setSplitValue(splitThreshold);

    var dfSplit = dataframe.split();
    int expectedTrainSize = (int) (splitThreshold * expectedRows);

    assertTrainTest(expectedRows, expectedCols, dfSplit, expectedTrainSize);
  }

  @Test
  public void must_load_csv_train_test_dataframe() throws IOException {
    var dataframe = Dataframes.csvTrainTest(getResourceFileName("org/rsultan/utils/example.csv"));
    var dfSplit = dataframe.split();

    int expectedRows = 6;
    int expectedCols = 5;
    int expectedTrainSize = (int) (dataframe.getSplitValue() * expectedRows);
    assertTrainTest(expectedRows, expectedCols, dfSplit, expectedTrainSize);

  }

  @Test
  public void must_shuffle_dataframe() throws IOException {
    var dataframe = Dataframes.csvTrainTest(getResourceFileName("org/rsultan/utils/example.csv"));
    var dfSplit = dataframe.shuffle();

    assertThat(dfSplit.getRowSize()).isEqualTo(6);
    assertThat(dfSplit.getColumnSize()).isEqualTo(5);
  }

  @Test
  public void must_shuffle_and_split_dataframe() throws IOException {
    var dataframe = Dataframes.csvTrainTest(getResourceFileName("org/rsultan/utils/example.csv"));
    var dfSplit = dataframe.shuffle().split();

    int expectedRows = 6;
    int expectedCols = 5;
    int expectedTrainSize = (int) (dataframe.getSplitValue() * expectedRows);
    assertTrainTest(expectedRows, expectedCols, dfSplit, expectedTrainSize);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_throw_IllegalArgumentException_when_split_value_is_wrong")
  public void must_throw_IllegalArgumentException_when_split_value_is_wrong(double splitValue) {
    assertThrows(IllegalArgumentException.class,
        () -> Dataframes.csvTrainTest(getResourceFileName("org/rsultan/utils/example.csv"))
            .setSplitValue(splitValue));
  }

  private void assertTrainTest(int expectedRows, int expectedCols, TrainTestSplit dfSplit,
      int expectedTrainSize) {
    assertThat(dfSplit.train().getRowSize()).isEqualTo(expectedTrainSize);
    assertThat(dfSplit.train().getColumnSize()).isEqualTo(expectedCols);

    assertThat(dfSplit.test().getRowSize()).isEqualTo(expectedRows - expectedTrainSize);
    assertThat(dfSplit.test().getColumnSize()).isEqualTo(expectedCols);
  }
}
