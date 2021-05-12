package org.rsultan.utils;

import static java.util.Collections.singletonList;
import static java.util.stream.IntStream.range;
import static java.util.stream.Stream.iterate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Objects;
import java.util.function.Function;
import java.util.regex.Pattern;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvFormat;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvParser;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvParserSettings;
import org.rsultan.dataframe.Column;

public class CSVUtils {

  private static final Pattern DOUBLE_VALUE_REGEX = Pattern.compile("-?\\d+\\.\\d+");
  private static final Pattern LONG_VALUE_REGEX = Pattern.compile("-?\\d+");
  private static final String HEADER_PREFIX = "c";

  private static Object getValueWithType(String value) {
    if (DOUBLE_VALUE_REGEX.matcher(value).matches()) {
      return Double.parseDouble(value);
    } else if (LONG_VALUE_REGEX.matcher(value).matches()) {
      return Long.valueOf(value);
    }
    return value;
  }

  public static Column<?>[] read(
      String fileName,
      String separator,
      String enclosure,
      boolean withHeader) throws IOException {
    var path = Paths.get(fileName);
    var parser = getParser(separator, enclosure);
    parser.beginParsing(path.toFile());

    var reader = Files.newBufferedReader(path);
    String[] firstLine = parser.parseNext();
    var columns = range(0, firstLine.length).boxed()
        .map(buildColumnHeaderName(withHeader, firstLine))
        .toArray(Column[]::new);

    iterate(0, i -> i + 1).map(i -> parser.parseNext())
        .takeWhile(Objects::nonNull)
        .forEach(lineArray ->
            range(0, lineArray.length).forEach(index -> {
              var typedValue = getValueWithType(lineArray[index]);
              columns[index].values().add(typedValue);
            }));
    reader.close();
    return columns;
  }

  private static CsvParser getParser(String separator, String enclosure) {
    CsvParserSettings settings = new CsvParserSettings();
    CsvFormat format = settings.getFormat();
    if (enclosure != null) {
      format.setQuote(enclosure.toCharArray()[0]);
    }
    format.setDelimiter(separator);
    format.setLineSeparator("\n");
    return new CsvParser(settings);
  }

  private static Function<Integer, Column<?>> buildColumnHeaderName(boolean withHeader,
      String[] firstLine) {
    return index -> {
      var firstLineCell = firstLine[index];
      return withHeader ?
          new Column<>(firstLineCell, new ArrayList<>()) :
          new Column<>(HEADER_PREFIX.concat(index.toString()),
              new ArrayList<>(singletonList((getValueWithType(firstLineCell)))));
    };
  }
}
