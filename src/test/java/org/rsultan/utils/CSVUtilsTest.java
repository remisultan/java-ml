package org.rsultan.utils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.of;
import static org.rsultan.utils.TestUtils.getResourceFileName;

import java.io.IOException;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

public class CSVUtilsTest {

  public static final String MALFORMED_EXAMPLED_CSV = "org/rsultan/utils/malformed_example.csv";
  public static final String EXAMPLE_CSV = "org/rsultan/utils/example.csv";
  public static final String EXAMPLE_CSV_NO_HEADER = "org/rsultan/utils/example_no_header.csv";

  private static Stream<Arguments> params_that_must_throw_exception_due_to_wrong_input() {
    return Stream.of(
        of(null, null, null, false, NullPointerException.class),
        of(null, null, null, true, NullPointerException.class),
        of(UUID.randomUUID().toString(), null, null, false, IllegalArgumentException.class),
        of(UUID.randomUUID().toString(), null, null, true, IllegalArgumentException.class),
        of(getResourceFileName(MALFORMED_EXAMPLED_CSV), null, null, false,
            IllegalArgumentException.class),
        of(getResourceFileName(MALFORMED_EXAMPLED_CSV), "\"", null, false,
            IllegalArgumentException.class)
    );
  }

  @ParameterizedTest
  @MethodSource("params_that_must_throw_exception_due_to_wrong_input")
  public void must_throw_exception_due_to_wrong_input(String fileName,
      String enclosures,
      String separator,
      boolean header,
      Class<? extends Exception> exception) {
    assertThrows(exception, () -> CSVUtils.read(fileName, separator, enclosures, header));
  }

  @Test
  public void must_read_csv_and_return_columns() throws IOException {
    var columns = CSVUtils.read(getResourceFileName(EXAMPLE_CSV), ",", "\"", true);

    assertThat(columns).hasSize((5));

    assertThat(columns[0].columnName()).isEqualTo("y");
    assertThat((List<Long>) columns[0].values()).hasSize(6)
        .containsExactly(1L, 2L, 3L, 4L, 5L, -6L);
    assertThat(columns[1].columnName()).isEqualTo("x");
    assertThat((List<Double>) columns[1].values()).hasSize(6)
        .containsExactly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D, -5.0D);
    assertThat(columns[2].columnName()).isEqualTo("x2");
    assertThat((List<Long>) columns[2].values()).hasSize(6)
        .containsExactly(1L, 4L, 9L, 16L, 25L, -25L);
    assertThat(columns[3].columnName()).isEqualTo("x3");
    assertThat((List<Long>) columns[3].values()).hasSize(6)
        .containsExactly(1L, 8L, 27L, 64L, 125L, -125L);
    assertThat(columns[4].columnName()).isEqualTo("strColumn");
    assertThat((List<String>) columns[4].values()).hasSize(6)
        .containsExactly("a", "b", "c", "d", "e", "f");

  }

  @Test
  public void must_read_csv_with_no_header_and_return_columns() throws IOException {
    var columns = CSVUtils.read(getResourceFileName(EXAMPLE_CSV_NO_HEADER), ",", "\"", false);

    assertThat(columns).hasSize(5);

    assertThat(columns[0].columnName()).isEqualTo("c0");
    assertThat((List<Long>) columns[0].values()).hasSize(5).containsExactly(1L, 2L, 3L, 4L, 5L);
    assertThat(columns[1].columnName()).isEqualTo("c1");
    assertThat((List<Double>) columns[1].values()).hasSize(5)
        .containsExactly(1.0D, 2.0D, 3.0D, 4.0D, 5.0D);
    assertThat(columns[2].columnName()).isEqualTo("c2");
    assertThat((List<Long>) columns[2].values()).hasSize(5).containsExactly(1L, 4L, 9L, 16L, 25L);
    assertThat(columns[3].columnName()).isEqualTo("c3");
    assertThat((List<Long>) columns[3].values()).hasSize(5).containsExactly(1L, 8L, 27L, 64L, 125L);
    assertThat(columns[4].columnName()).isEqualTo("c4");
    assertThat((List<String>) columns[4].values()).hasSize(5)
        .containsExactly("a", "b", "c", "d", "e");
  }
}
