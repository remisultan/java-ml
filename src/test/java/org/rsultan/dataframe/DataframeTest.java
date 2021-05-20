package org.rsultan.dataframe;

import static java.util.stream.IntStream.range;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

public class DataframeTest {

  private static Stream<Arguments> params_that_must_load_dataframe_correctly() {
    return Stream.of(
        of(new Column[]{new Column<>("Integers", 0, 1, 2, 3, 4)}, 5, 1),
        of(new Column[]{new Column<>("Longs", 0L, 1L, 2L, 3L, 4L)}, 5, 1),
        of(new Column[]{new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D)}, 5, 1),
        of(new Column[]{new Column<>("Floats", 0F, 1F, 2F, 3F, 4F)}, 5, 1),
        of(new Column[]{new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3")}, 5, 1),
        of(new Column[]{new Column<>("Negatives", "-1.1", "-2.1", "-3", "-4.4", "-5.3")}, 5, 1),
        of(new Column<?>[]{
            new Column<>("Integers", 0, 1, 2, 3, 4),
            new Column<>("Longs", 0L, 1L, 2L, 3L, 4L),
            new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D),
            new Column<>("Floats", 0F, 1F, 2F, 3F, 4F),
            new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3"),
            new Column<>("Negatives", "-1.1", "-2.1", "-3", "-4.4", "-5.3")
        }, 5, 6)
    );
  }

  private static Stream<Arguments> params_that_must_throw_exception_due_to_malformed_input() {
    return Stream.of(
        of(new Column[]{new Column<>(null, (List<Object>) null)}, NullPointerException.class),
        of(new Column[]{new Column<>(null, 0, 2, 3, 4)}, NullPointerException.class),
        of(new Column<?>[]{new Column<>("c1", 1, 2), new Column<>("c2", 1, 2, 3)},
            IllegalArgumentException.class),
        of(new Column<?>[]{new Column<>("c1", 1, "lat65", 3), new Column<>("c2", 1, 2, 3)},
            IllegalArgumentException.class)
    );
  }

  private static Stream<Arguments> params_that_must_throw_exception_due_to_malformed_row_input() {
    return Stream.of(
        of(null, null, NullPointerException.class),
        of(new String[]{"c1", "c2", "c3", "c4", "c5"}, null, NullPointerException.class),
        of(new String[]{"c1", "c2", "c3", "c4", "c5"}, new Row[]{new Row(1, 3, 3)},
            IllegalArgumentException.class),
        of(new String[]{"c1", "c2", "c3", "c4", "c5"},
            new Row[]{new Row(1, 3, 3), new Row(1, 3, 3, 4)},
            IllegalArgumentException.class)
    );
  }

  @Test
  public void must_load_dataframe_correctly_with_empty_dataframe() {
    var df = Dataframes.create();
    assertThat(df.getRowSize()).isEqualTo(0);
    assertThat(df.getColumnSize()).isEqualTo(0);
  }

  @Test
  public void must_one_hot_encode_column() {
    var df = Dataframes
        .create(new Column<>("colors", List.of("red", "green", "blue", "yellow")))
        .oneHotEncode("colors");

    var matrix = df.toMatrix("red", "green", "blue", "yellow");

    assertThat(df.getRowSize()).isEqualTo(4);
    assertThat(df.getColumnSize()).isEqualTo(5);
    assertThat(df.get("red")).containsExactly(true, false, false, false);
    assertThat(df.get("green")).containsExactly(false, true, false, false);
    assertThat(df.get("blue")).containsExactly(false, false, true, false);
    assertThat(df.get("yellow")).containsExactly(false, false, false, true);

    assertThat(matrix.getColumn(0).toDoubleVector()).containsExactly(1, 0, 0, 0);
    assertThat(matrix.getColumn(1).toDoubleVector()).containsExactly(0, 1, 0, 0);
    assertThat(matrix.getColumn(2).toDoubleVector()).containsExactly(0, 0, 1, 0);
    assertThat(matrix.getColumn(3).toDoubleVector()).containsExactly(0, 0, 0, 1);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_dataframe_correctly")
  public void must_load_dataframe_correctly(Column<?>[] columns, int expectedRows,
      int expectedCols) {
    var dataframe = Dataframes.create(columns);

    assertThat(dataframe.getRowSize()).isEqualTo(expectedRows);
    assertThat(dataframe.getColumnSize()).isEqualTo(expectedCols);
    var matrix = dataframe
        .toMatrix(Stream.of(columns).map(Column::columnName).toArray(String[]::new));
    range(0, columns.length).forEach(idx -> {
      var column = columns[idx];
      var actualValues = dataframe.get(column.columnName());
      var expectedValues = columns[idx].values().toArray();
      var vector = dataframe.toVector(column.columnName());
      var expectedValuesArray = Stream.of(expectedValues)
          .map(String::valueOf)
          .mapToDouble(Double::parseDouble)
          .toArray();

      assertThat(actualValues).containsExactly(expectedValues);
      assertThat(vector.toDoubleVector()).containsExactly(expectedValuesArray);
      assertThat(vector.toDoubleVector()).containsExactly(matrix.getColumn(idx).toDoubleVector());
    });
    dataframe.show(expectedRows);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_dataframe_correctly")
  public void must_load_train_test_dataframe_correctly(Column<?>[] columns, int expectedRows,
      int expectedCols) {
    var dataframe = Dataframes.create(columns);

    assertThat(dataframe.getRowSize()).isEqualTo(expectedRows);
    assertThat(dataframe.getColumnSize()).isEqualTo(expectedCols);
    var matrix = dataframe
        .toMatrix(Stream.of(columns).map(Column::columnName).toArray(String[]::new));
    range(0, columns.length).forEach(idx -> {
      var column = columns[idx];
      var actualValues = dataframe.get(column.columnName());
      var expectedValues = columns[idx].values().toArray();
      var vector = dataframe.toVector(column.columnName());
      var expectedValuesArray = Stream.of(expectedValues)
          .map(String::valueOf)
          .mapToDouble(Double::parseDouble)
          .toArray();

      assertThat(actualValues).containsExactly(expectedValues);
      assertThat(vector.toDoubleVector()).containsExactly(expectedValuesArray);
      assertThat(vector.toDoubleVector()).containsExactly(matrix.getColumn(idx).toDoubleVector());
    });
    dataframe.show(expectedRows);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_throw_exception_due_to_malformed_input")
  public void must_throw_exception_due_to_malformed_input(Column<?>[] columns,
      Class<? extends Exception> exceptionClass) {
    assertThrows(exceptionClass, () -> {
      Dataframe dataframe = Dataframes.create(columns);
      dataframe.show(10);
      dataframe.toMatrix();
    });
  }

  @ParameterizedTest
  @MethodSource("params_that_must_throw_exception_due_to_malformed_row_input")
  public void must_throw_exception_due_to_malformed_row_input(
      String[] columnNames,
      Row[] rows,
      Class<? extends Exception> exceptionClass) {
    assertThrows(exceptionClass, () -> {
      Dataframe dataframe = Dataframes.create(columnNames, rows);
      dataframe.show(10);
      dataframe.toMatrix();
    });
  }

  @Test
  public void must_create_new_column() {
    var df = Dataframes.create(new Column<>("doubles", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D));
    df = df.map("ones", () -> 1);

    assertThat(df.get("ones")).containsExactly(1, 1, 1, 1, 1);
  }

  @Test
  public void must_create_new_column_from_existing_one() {
    var df = Dataframes.create(new Column<>("doubles", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D));
    df = df.map("exp", Math::exp, "doubles");

    assertThat(df.get("exp"))
        .containsExactly(Math.exp(1.0D), Math.exp(2.0D), Math.exp(3.0D), Math.exp(4.0D),
            Math.exp(5.0D));
  }

  @Test
  public void must_create_new_column_from_two_existing_cols() {
    var df = Dataframes.create(
        new Column<>("d1", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D),
        new Column<>("d2", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D)
    );
    df = df.map("square", (Double d1, Double d2) -> d1 * d2, "d1", "d2");

    assertThat(df.get("square")).containsExactly(1.0D, 4.0D, 9.0D, 16.0D, 25.0D);
  }

  @Test
  public void must_filter_with_predicate() {
    var df = Dataframes.create(
        new Column<>("d1", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D),
        new Column<>("d2", 1.0D, 5.0D, 7.0D, 9.0D, 11.0D)
    );
    df = df.filter("d1", (Double d1) -> d1 % 2 == 0);

    assertThat(df.getRowSize()).isEqualTo(2);
    assertThat(df.get("d1")).containsExactly(2.0D, 4.0D);
    assertThat(df.get("d2")).containsExactly(5.0D, 9.0D);
  }

  @Test
  public void must_filter_with_bipredicate() {
    var df = Dataframes.create(
        new Column<>("d1", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D),
        new Column<>("d2", 1.0D, 5.0D, 7.0D, 9.0D, 11.0D)
    );
    df = df.filter("d1", "d2", (Double d1, Double d2) -> d1 * d2 > 20D);

    assertThat(df.getRowSize()).isEqualTo(3);
    assertThat(df.get("d1")).containsExactly(3.0D, 4.0, 5.0D);
    assertThat(df.get("d2")).containsExactly(7.0D, 9.0D, 11.0D);
  }

  @Test
  public void must_remove_column() {
    var df = Dataframes.create(
        new Column<>("d1", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D),
        new Column<>("d2", 1.0D, 5.0D, 7.0D, 9.0D, 11.0D)
    );
    df = df.mapWithout("d1");

    assertThat(df.getColumnSize()).isEqualTo(1);
    assertThat(df.get("d2")).containsExactly(1.0D, 5.0D, 7.0D, 9.0D, 11.0D);
  }

  @Test
  public void must_load_dataframe_from_csv() throws IOException {
    var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example.csv"));
    assertThat(df.get("y")).containsExactly(1L, 2L, 3L, 4L, 5L, -6L);
    assertThat(df.get("x")).containsExactly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D, -5.0D);
    assertThat(df.get("x2")).containsExactly(1L, 4L, 9L, 16L, 25L, -25L);
    assertThat(df.get("x3")).containsExactly(1L, 8L, 27L, 64L, 125L, -125L);
    assertThat(df.get("strColumn")).containsExactly("a", "b", "c", "d", "e", "f");
  }

  @Test
  public void must_load_dataframe_from_csv_with_no_header() throws IOException {
    var df = Dataframes
        .csv(getResourceFileName("org/rsultan/utils/example_no_header.csv"), ",", "\"", false);
    assertThat(df.get("c0")).containsExactly(1L, 2L, 3L, 4L, 5L);
    assertThat(df.get("c1")).containsExactly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D);
    assertThat(df.get("c2")).containsExactly(1L, 4L, 9L, 16L, 25L);
    assertThat(df.get("c3")).containsExactly(1L, 8L, 27L, 64L, 125L);
    assertThat(df.get("c4")).containsExactly("a", "b", "c", "d", "e");
  }
}
