package org.rsultan.dataframe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.engine.label.LabelValueIndexer;
import org.rsultan.dataframe.engine.mapper.impl.group.Aggregation;
import org.rsultan.dataframe.engine.mapper.impl.group.AggregationType;

public class DataframeTest {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  private static Stream<Arguments> params_that_must_load_dataframe_correctly() {
    return Stream.of(
        of(new String[]{"Integers"},
            List.of(List.of(0), List.of(1), List.of(2), List.of(3), List.of(4)), 5, 1),
        of(new String[]{"Longs"},
            List.of(List.of(0L), List.of(1L), List.of(2L), List.of(3L), List.of(4L)), 5, 1),
        of(new String[]{"Doubles"},
            List.of(List.of(0D), List.of(1D), List.of(2D), List.of(3D), List.of(4D)), 5, 1),
        of(new String[]{"Floats"},
            List.of(List.of(0F), List.of(1F), List.of(2F), List.of(3F), List.of(4F)), 5, 1),
        of(new String[]{"Strings"},
            List.of(List.of("1.1"), List.of("2.1"), List.of("3"), List.of("4.4"), List.of("5.3")),
            5, 1),
        of(new String[]{"Negatives"},
            List.of(List.of("-1.1"), List.of("-2.1"), List.of("-3"), List.of("-4.4"),
                List.of("-5.3")), 5, 1),
        of(new String[]{"Integers", "Longs", "Doubles", "Floats", "Strings", "Negatives",
                "Decimals", "Negative decimals", "Scientific"},
            List.of(
                List.of(0, 0L, 0D, 0F, "0", "-0", "0.0", "-0.0", "+0.0E10"),
                List.of(1, 1L, 1D, 1F, "1", "-1", "1.1", "-1.1", "1.1E10"),
                List.of(2, 2L, 2D, 2F, "2", "-2", "2.2", "-2.2", "-2.2E10"),
                List.of(3, 3L, 3D, 3F, "3", "-3", "3.3", "-3.3", "3.3E-10"),
                List.of(4, 4L, 4D, 4F, "4", "-4", "4.4", "-4.4", "+4.4E+10"),
                List.of(5, 5L, 5D, 5F, "5", "-5", "5.5", "-5.5", "5.5E10")
            )
            , 6, 9)
    );
  }

  private static Stream<Arguments> params_that_must_throw_exception_due_to_malformed_input() {
    return Stream.of(
        of(null, List.of(), IllegalArgumentException.class),
        of(new String[]{}, null, IllegalArgumentException.class),
        of(new String[]{}, getObjects(), IllegalArgumentException.class)
    );
  }

  private static ArrayList<Object> getObjects() {
    final ArrayList<Object> objects = new ArrayList<>();
    objects.add(null);
    return objects;
  }

  @Test
  public void must_load_dataframe_correctly_with_empty_dataframe() {
    var df = Dataframes.create(new String[]{}, List.of());
    var result = df.getResult();
    assertThat(result.header().size()).isEqualTo(0);
    assertThat(result.rows().size()).isEqualTo(0);
  }

  @Test
  public void must_one_hot_encode_column() {
    var df = Dataframes
        .create(new String[]{"colors"},
            List.of(List.of("red"), List.of("green"), List.of("blue"), List.of("yellow")))
        .oneHotEncode("colors")
        .mapWithout("colors");

    var result = df.getResult();
    var matrix = df.toMatrix();

    assertThat(result.header().size()).isEqualTo(4);
    assertThat(result.rows().size()).isEqualTo(4);
    assertThat(df.<Double>getColumn("blue")).containsExactly(0D, 0D, 1D, 0D);
    assertThat(df.<Double>getColumn("green")).containsExactly(0D, 1D, 0D, 0D);
    assertThat(df.<Double>getColumn("red")).containsExactly(1D, 0D, 0D, 0D);
    assertThat(df.<Double>getColumn("yellow")).containsExactly(0D, 0D, 0D, 1D);

    assertThat(matrix.getColumn(0).toDoubleVector()).containsExactly(0D, 0D, 1D, 0D);
    assertThat(matrix.getColumn(1).toDoubleVector()).containsExactly(0D, 1D, 0D, 0D);
    assertThat(matrix.getColumn(2).toDoubleVector()).containsExactly(1D, 0D, 0D, 0D);
    assertThat(matrix.getColumn(3).toDoubleVector()).containsExactly(0D, 0D, 0D, 1D);
  }

  @Test
  public void must_select_columns() {
    var df = Dataframes
        .create(new String[]{"colors"},
            List.of(List.of("red"), List.of("green"), List.of("blue"), List.of("yellow")))
        .oneHotEncode("colors")
        .mapWithout("colors")
        .select("blue", "yellow");

    var result = df.copy().getResult();
    assertThat(result.header().size()).isEqualTo(2);
    assertThat(result.rows().size()).isEqualTo(4);

    assertThat(df.copy().<Double>getColumn("blue")).containsExactly(0D, 0D, 1D, 0D);
    assertThat(df.copy().<Double>getColumn("yellow")).containsExactly(0D, 0D, 0D, 1D);

    var matrix = df.copy().toMatrix();
    assertThat(matrix.getColumn(0).toDoubleVector()).containsExactly(0D, 0D, 1D, 0D);
    assertThat(matrix.getColumn(1).toDoubleVector()).containsExactly(0D, 0D, 0D, 1D);
  }

  @Test
  public void must_label_value_columns() {
    var df = Dataframes
        .create(new String[]{"colors"},
            List.of(List.of("red"), List.of("green"), List.of("blue"), List.of("yellow"),
                List.of("purple")));

    var labelValueIndexer = new LabelValueIndexer<>("red", "green", "blue", "yellow");
    var matrix = df.toMatrix(Map.of("colors", labelValueIndexer));
    final double[] actual = matrix.getColumn(0).toDoubleVector();

    assertThat(actual).containsExactly(2.0, 1.0, 0.0, 3.0, Long.MIN_VALUE);
    assertThat(labelValueIndexer.getLabelValue(actual[0])).isEqualTo("red");
    assertThat(labelValueIndexer.getLabelValue(actual[1])).isEqualTo("green");
    assertThat(labelValueIndexer.getLabelValue(actual[2])).isEqualTo("blue");
    assertThat(labelValueIndexer.getLabelValue(actual[3])).isEqualTo("yellow");
    assertThat(labelValueIndexer.getLabelValue(actual[4])).isNull();

  }

  @Test
  public void must_label_value_columns_and_split() {
    var df = Dataframes
        .create(new String[]{"colors"},
            List.of(List.of("red"), List.of("green"), List.of("blue"), List.of("yellow"),
                List.of("purple")));

    var labelValueIndexer = new LabelValueIndexer<>("red", "green", "blue", "yellow");
    var matrices = df.trainTest(0.5D, Map.of("colors", labelValueIndexer));
    final double[] actual = matrices[0].getColumn(0).toDoubleVector();

    assertThat(actual).containsExactly(2.0, 1.0);
    assertThat(labelValueIndexer.getLabelValue(actual[0])).isEqualTo("red");
    assertThat(labelValueIndexer.getLabelValue(actual[1])).isEqualTo("green");

    final double[] actualTest = matrices[1].getColumn(0).toDoubleVector();

    assertThat(actualTest).containsExactly(0.0, 3.0, Long.MIN_VALUE);
    assertThat(labelValueIndexer.getLabelValue(actualTest[0])).isEqualTo("blue");
    assertThat(labelValueIndexer.getLabelValue(actualTest[1])).isEqualTo("yellow");
    assertThat(labelValueIndexer.getLabelValue(actualTest[2])).isNull();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_dataframe_correctly")
  public void must_load_dataframe_correctly(String[] header, List<List<?>> rows,
      int expectedRows,
      int expectedCols) {
    var dataframe = Dataframes.create(header, rows);
    var result = dataframe.getResult();

    assertThat(result.rows().size()).isEqualTo(expectedRows);
    assertThat(result.header().size()).isEqualTo(expectedCols);

    dataframe.show(10);

    var matrix = dataframe.toMatrix();
    System.out.println(matrix);
    assertThat(matrix.rows()).isEqualTo(expectedRows);
    assertThat(matrix.columns()).isEqualTo(expectedCols);
  }

  @ParameterizedTest
  @MethodSource("params_that_must_load_dataframe_correctly")
  public void must_train_test_dataframe(String[] header, List<List<?>> rows,
      int expectedRows,
      int expectedCols) {
    var dataframe = Dataframes.create(header, rows);
    var result = dataframe.getResult();

    assertThat(result.rows().size()).isEqualTo(expectedRows);
    assertThat(result.header().size()).isEqualTo(expectedCols);

    dataframe.show(10);

    var matrices = dataframe.trainTest(0.5D);
    System.out.println(matrices[0]);
    assertThat(matrices[0].rows()).isEqualTo((int) (expectedRows * 0.5));
    assertThat(matrices[0].columns()).isEqualTo(expectedCols);
    System.out.println(matrices[1]);
    assertThat(matrices[1].rows()).isEqualTo(expectedRows - (int) (expectedRows * 0.5));
    assertThat(matrices[1].columns()).isEqualTo(expectedCols);
  }

  @Test
  public void must_write_dataframe_to_file() throws IOException {
    var headers = new String[]{"c1", "c2", "c3"};
    final String property = System.getProperty("java.io.tmpdir");
    String filename =
        (property.endsWith("/") ? property : property + "/") + UUID.randomUUID() + ".csv";

    Dataframes.create(headers,
        List.of(
            List.of("1", 2, 3.0D),
            List.of("4", "5", 6),
            List.of(7, 8, 9)
        )
    ).write(filename, ",", "\"");

    File file = new File(filename);
    assertThat(Files.exists(file.toPath())).isTrue();
  }

  @ParameterizedTest
  @MethodSource("params_that_must_throw_exception_due_to_malformed_input")
  public void must_throw_exception_due_to_malformed_input(String[] header, List<List<?>> rows,
      Class<? extends Exception> exceptionClass) {
    assertThrows(exceptionClass, () -> {
      Dataframe dataframe = Dataframes.create(header, rows);
      dataframe.show(10);
      dataframe.toMatrix();
    });
  }

  @Test
  public void must_create_new_column_from_existing_one() {
    var df = Dataframes
        .create(new String[]{"doubles"},
            List.of(List.of(1.0D), List.of(2.0D), List.of(3.0D), List.of(4.0D), List.of(5.0D)))
        .map("exp", Math::exp, "doubles");

    assertThat(df.getColumn("exp"))
        .containsExactly(Math.exp(1.0D), Math.exp(2.0D), Math.exp(3.0D), Math.exp(4.0D),
            Math.exp(5.0D));
  }

  @Test
  public void must_transform_column() {
    var df = Dataframes
        .create(new String[]{"doubles"},
            List.of(List.of(1.0D), List.of(2.0D), List.of(3.0D), List.of(4.0D), List.of(5.0D)))
        .transform("doubles", Math::exp);

    assertThat(df.getColumn("doubles"))
        .containsExactly(Math.exp(1.0D), Math.exp(2.0D), Math.exp(3.0D), Math.exp(4.0D),
            Math.exp(5.0D));
  }

  @Test
  public void must_transform_column_from_two_existing_cols() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 2.0D),
        List.of(3.0D, 3.0D),
        List.of(4.0D, 4.0D),
        List.of(5.0D, 5.0D)
    )).transform("d1", "d2", (Double d1, Double d2) -> d1 * d2);

    assertThat(df.getColumn("d1")).containsExactly(1.0D, 4.0D, 9.0D, 16.0D, 25.0D);
  }

  @Test
  public void must_create_new_column_from_two_existing_cols() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 2.0D),
        List.of(3.0D, 3.0D),
        List.of(4.0D, 4.0D),
        List.of(5.0D, 5.0D)
    ));
    df = df.map("square", (Double d1, Double d2) -> d1 * d2, "d1", "d2");

    assertThat(df.getColumn("square")).containsExactly(1.0D, 4.0D, 9.0D, 16.0D, 25.0D);
  }

  @Test
  public void must_filter_with_predicate() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 5.0D),
        List.of(3.0D, 7.0D),
        List.of(4.0D, 9.0D),
        List.of(5.0D, 11.0D)
    )).filter("d1", (Double d1) -> d1 % 2 == 0);

    assertThat(df.getColumn("d1")).containsExactly(2.0D, 4.0D);
    assertThat(df.getColumn("d2")).containsExactly(5.0D, 9.0D);
  }

  @Test
  public void must_filter_with_bipredicate() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 5.0D),
        List.of(3.0D, 7.0D),
        List.of(4.0D, 9.0D),
        List.of(5.0D, 11.0D)
    )).filter("d1", "d2", (Double d1, Double d2) -> d1 * d2 > 20D);

    assertThat(df.getColumn("d1")).containsExactly(3.0D, 4.0, 5.0D);
    assertThat(df.getColumn("d2")).containsExactly(7.0D, 9.0D, 11.0D);
  }

  @Test
  public void must_supply_value() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
            List.of(1.0D, 1.0D),
            List.of(2.0D, 5.0D),
            List.of(3.0D, 7.0D),
            List.of(4.0D, 9.0D),
            List.of(5.0D, 11.0D)
        ))
        .map("generated", () -> 1)
        .select("generated");

    assertThat(df.getColumn("generated")).containsExactly(1, 1, 1, 1, 1);
  }

  @Test
  public void must_add_column_from_list() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
            List.of(1.0D, 1.0D),
            List.of(2.0D, 5.0D),
            List.of(3.0D, 7.0D),
            List.of(4.0D, 9.0D),
            List.of(5.0D, 11.0D)
        ))
        .addColumn("generated", List.of(1.0D, 4.0D, 6.0D, 8.0D, 10.0D, 12.0D))
        .select("generated");

    assertThat(df.getColumn("generated")).containsExactly(1.0D, 4.0D, 6.0D, 8.0D, 10.0D);
  }

  @Test
  public void must_add_column_from_list_with_null_values() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
            List.of(1.0D, 1.0D),
            List.of(2.0D, 5.0D),
            List.of(3.0D, 7.0D),
            List.of(4.0D, 9.0D),
            List.of(5.0D, 11.0D)
        ))
        .addColumn("generated", List.of(1.0D, 4.0D, 6.0D, 8.0D))
        .select("generated");

    final List<Object> generated = df.copy().getColumn("generated");
    assertThat(generated).containsExactly(1.0D, 4.0D, 6.0D, 8.0D, null);
  }

  @Test
  public void must_remove_column() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 5.0D),
        List.of(3.0D, 7.0D),
        List.of(4.0D, 9.0D),
        List.of(5.0D, 11.0D)
    )).mapWithout("d1");

    var result = df.getResult();
    assertThat(result.header().size()).isEqualTo(1);
    assertThat(result.rows().size()).isEqualTo(5);
    assertThat(df.copy().getColumn("d2")).containsExactly(1.0D, 5.0D, 7.0D, 9.0D, 11.0D);
  }

  @Test
  public void must_shuffle() {
    var df = Dataframes.create(new String[]{"d1", "d2"}, List.of(
        List.of(1.0D, 1.0D),
        List.of(2.0D, 5.0D),
        List.of(3.0D, 7.0D),
        List.of(4.0D, 9.0D),
        List.of(5.0D, 11.0D)
    )).shuffle();

    var result = df.getResult();
    assertThat(result.header().size()).isEqualTo(2);
    assertThat(result.rows().size()).isEqualTo(5);
    assertThat(df.getColumn("d1")).containsOnly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D);
    assertThat(df.getColumn("d2")).containsOnly(1.0D, 5.0D, 7.0D, 9.0D, 11.0D);
  }

  @Test
  public void must_load_dataframe_from_csv() throws IOException {
    var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example.csv"));
    df.getResult();

    assertThat(df.getColumn("y")).containsExactly(1L, 2L, 3L, 4L, 5L, -6L);
    assertThat(df.getColumn("x")).containsExactly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D, -5.0D);
    assertThat(df.getColumn("x2")).containsExactly(1L, 4L, 9L, 16L, 25L, -25L);
    assertThat(df.getColumn("x3")).containsExactly(1L, 8L, 27L, 64L, 125L, -125L);
    assertThat(df.getColumn("strColumn")).containsExactly("a", "b", "c", "d", "e", "f");
  }

  @Test
  public void must_group_by() throws IOException {
    var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example-classif.csv"))
        .groupBy("strColumn",
            new Aggregation("y", AggregationType.SUM),
            new Aggregation("y", AggregationType.AVG),
            new Aggregation("y", AggregationType.COUNT),
            new Aggregation("y", AggregationType.MAX),
            new Aggregation("y", AggregationType.MIN),
            new Aggregation("y", AggregationType.ACCUMULATE));
    df.getResult();
    df.show(10);
    assertThat(df.getColumn("sum(y)")).containsExactly(3.0, 7.0, 5.0);
    assertThat(df.getColumn("avg(y)")).containsExactly(1.5, 3.5, 5.0);
    assertThat(df.getColumn("count(y)")).containsExactly(2L, 2L, 1L);
    assertThat(df.getColumn("max(y)")).containsExactly(2.0D, 4.0D, 5.0D);
    assertThat(df.getColumn("min(y)")).containsExactly(1.0D, 3.0D, 5.0D);
    assertThat(df.getColumn("accumulate(y)")).containsExactly(
        List.of(1L, 2L), List.of(3L, 4L), List.of(5L));
  }
}
