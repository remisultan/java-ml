package org.rsultan.dataframe;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.rsultan.utils.CSVUtilsTest;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.IntStream.range;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;

public class DataframeTest {

    private static Stream<Arguments> params_that_must_load_dataframe_correctly() {
        return Stream.of(
                of(new Column[]{new Column<>("Integers", 0, 1, 2, 3, 4)}, 5, 1),
                of(new Column[]{new Column<>("Longs", 0L, 1L, 2L, 3L, 4L)}, 5, 1),
                of(new Column[]{new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D)}, 5, 1),
                of(new Column[]{new Column<>("Floats", 0F, 1F, 2F, 3F, 4F)}, 5, 1),
                of(new Column[]{new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3")}, 5, 1),
                of(new Column<?>[]{
                        new Column<>("Integers", 0, 1, 2, 3, 4),
                        new Column<>("Longs", 0L, 1L, 2L, 3L, 4L),
                        new Column<>("Doubles", 0D, 1D, 2D, 3D, 4D),
                        new Column<>("Floats", 0F, 1F, 2F, 3F, 4F),
                        new Column<>("Strings", "1.1", "2.1", "3", "4.4", "5.3")
                }, 5, 5)
        );
    }

    private static Stream<Arguments> params_that_must_throw_exception_due_to_malformed_input() {
        return Stream.of(
                of(new Column[]{new Column<>(null, (List<Object>) null)}, NullPointerException.class),
                of(new Column[]{new Column<>(null, 0, 2, 3, 4)}, NullPointerException.class),
                of(new Column<?>[]{new Column<>("c1", 1, 2), new Column<>("c2", 1, 2, 3)}, IllegalArgumentException.class)
        );
    }

    public static String getResourceFileName(String resourcePath) {
        var classLoader = CSVUtilsTest.class.getClassLoader();
        return new File(classLoader.getResource(resourcePath).getFile()).toString();
    }

    @Test
    public void must_load_dataframe_correctly_with_empty_dataframe() {
        var df = Dataframes.create();
        assertThat(df.getRows()).isEqualTo(0);
        assertThat(df.getColumns()).isEqualTo(0);
    }

    @ParameterizedTest
    @MethodSource("params_that_must_load_dataframe_correctly")
    public void must_load_dataframe_correctly(Column<?>[] columns, int expectedRows, int expectedCols) {
        var dataframe = Dataframes.create(columns);

        assertThat(dataframe.getRows()).isEqualTo(expectedRows);
        assertThat(dataframe.getColumns()).isEqualTo(expectedCols);
        var matrix = dataframe.toMatrix(Stream.of(columns).map(Column::columnName).toArray(String[]::new));
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
    public void must_throw_exception_due_to_malformed_input(Column<?>[] columns, Class<? extends Exception> exceptionClass) {
        assertThrows(exceptionClass, () -> Dataframes.create(columns).show(10));
    }

    @Test
    public void must_create_new_column() {
        var df = Dataframes.create(new Column<>("doubles", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D));
        df = df.withColumn("ones", () -> 1);

        assertThat(df.get("ones")).containsExactly(1, 1, 1, 1, 1);
    }

    @Test
    public void must_create_new_column_from_existing_one() {
        var df = Dataframes.create(new Column<>("doubles", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D));
        df = df.withColumn("exp", "doubles", Math::exp);

        assertThat(df.get("exp")).containsExactly(Math.exp(1.0D), Math.exp(2.0D), Math.exp(3.0D), Math.exp(4.0D), Math.exp(5.0D));
    }

    @Test
    public void must_create_new_column_from_two_existing_cols() {
        var df = Dataframes.create(
                new Column<>("d1", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D),
                new Column<>("d2", 1.0D, 2.0D, 3.0D, 4.0D, 5.0D)
        );
        df = df.withColumn("square", (Double d1, Double d2) -> d1 * d2, "d1", "d2");

        assertThat(df.get("square")).containsExactly(1.0D, 4.0D, 9.0D, 16.0D, 25.0D);
    }

    @Test
    public void must_load_dataframe_from_csv() throws IOException {
        var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example.csv"));
        assertThat(df.get("y")).containsExactly(1L, 2L, 3L, 4L);
        assertThat(df.get("x")).containsExactly(1.0D, 2.0D, 3.0D, 4.0D);
        assertThat(df.get("x2")).containsExactly(1L, 4L, 9L, 16L);
        assertThat(df.get("x3")).containsExactly(1L, 8L, 27L, 64L);
        assertThat(df.get("strColumn")).containsExactly("a", "b", "c", "d");
    }

    @Test
    public void must_load_dataframe_from_csv_with_no_header() throws IOException {
        var df = Dataframes.csv(getResourceFileName("org/rsultan/utils/example_no_header.csv"), ",", false);
        assertThat(df.get("c0")).containsExactly(1L, 2L, 3L, 4L);
        assertThat(df.get("c1")).containsExactly(1.0D, 2.0D, 3.0D, 4.0D);
        assertThat(df.get("c2")).containsExactly(1L, 4L, 9L, 16L);
        assertThat(df.get("c3")).containsExactly(1L, 8L, 27L, 64L);
        assertThat(df.get("c4")).containsExactly("a", "b", "c", "d");
    }
}
