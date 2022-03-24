package org.rsultan.dataframe.engine.source.impl;

import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvFormat;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvParser;
import org.junit.jupiter.params.shadow.com.univocity.parsers.csv.CsvParserSettings;
import org.rsultan.dataframe.Row;

import java.nio.file.Paths;
import java.util.regex.Pattern;
import org.rsultan.dataframe.engine.source.SourceDataProcessor;

import static java.lang.String.format;
import static java.util.Arrays.asList;
import static java.util.Arrays.stream;
import static java.util.Objects.isNull;
import static java.util.Objects.nonNull;
import static java.util.Optional.ofNullable;
import static java.util.regex.Pattern.compile;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.rsultan.utils.CSVUtils.COLUMN_VALUE_GROUP_NAME;
import static org.rsultan.utils.CSVUtils.DOUBLE_VALUE_REGEX;
import static org.rsultan.utils.CSVUtils.HEADER_PREFIX;
import static org.rsultan.utils.CSVUtils.LONG_VALUE_REGEX;
import static org.rsultan.utils.CSVUtils.TRIM_ENCLOSURE_PATTERN;
import static org.rsultan.utils.Constants.NULL_ROW;

public class CsvFileSource extends SourceDataProcessor {

  private final String fileName;
  private final String separator;
  private final String enclosure;
  private final boolean withHeader;

  private Pattern enclosurePattern;
  private CsvParser parser;

  public CsvFileSource(String fileName, String separator, String enclosure, boolean withHeader) {
    super();
    this.fileName = fileName;
    this.separator = separator;
    this.enclosure = enclosure;
    this.withHeader = withHeader;
  }

  @Override
  public Row produce() {
    final String[] lineArray = parser.parseNext();
    if (ofNullable(lineArray).isEmpty()) {
      return NULL_ROW;
    }
    var values = stream(lineArray).map(this::trimEnclosures).map(CsvFileSource::getValueWithType)
        .collect(toList());
    return new Row(values);
  }

  @Override
  public void run() {
    var path = Paths.get(fileName);
    parser = getParser(separator, enclosure, withHeader);
    parser.beginParsing(path.toFile());
    enclosurePattern = getEnclosurePattern();
    header = withHeader ? asList(parser.getContext().headers())
        : range(0, parser.parseNext().length).mapToObj(i -> HEADER_PREFIX + i).collect(toList());
    super.run();
  }

  @Override
  protected boolean canStop(Row row) {
    return NULL_ROW.equals(row);
  }

  private Pattern getEnclosurePattern() {
    return isNull(enclosure) ? null : compile(
        format(TRIM_ENCLOSURE_PATTERN, enclosure, enclosure, enclosure, enclosure, enclosure));
  }

  private String trimEnclosures(String value) {
    if (nonNull(enclosurePattern)) {
      var matcher = enclosurePattern.matcher(value);
      if (matcher.matches()) {
        return matcher.group(COLUMN_VALUE_GROUP_NAME);
      }
    }
    return value;
  }

  private static CsvParser getParser(String separator, String enclosure, boolean withHeader) {
    CsvParserSettings settings = new CsvParserSettings();
    settings.setHeaderExtractionEnabled(withHeader);
    CsvFormat format = settings.getFormat();
    if (enclosure != null) {
      format.setQuote(enclosure.toCharArray()[0]);
    }
    format.setDelimiter(separator);
    format.setLineSeparator("\n");
    return new CsvParser(settings);
  }

  private static Object getValueWithType(String value) {
    if (DOUBLE_VALUE_REGEX.matcher(value).matches()) {
      return Double.parseDouble(value);
    } else if (LONG_VALUE_REGEX.matcher(value).matches()) {
      return Long.valueOf(value);
    }
    return value;
  }
}
