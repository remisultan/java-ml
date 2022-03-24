package org.rsultan.dataframe;

import java.io.IOException;
import java.util.List;
import org.rsultan.dataframe.engine.source.impl.CsvFileSource;
import org.rsultan.dataframe.engine.source.impl.InlineSource;

public class Dataframes {

  public static Dataframe create(String[] columnNames, List<List<?>> rows) {
    return new Dataframe(
        InlineSource.class,
        columnNames,
        rows
    );
  }

  public static Dataframe csv(String fileName) throws IOException {
    return csv(fileName, ",");
  }

  public static Dataframe csv(String fileName, String separator) throws IOException {
    return csv(fileName, separator, "\"");
  }

  public static Dataframe csv(String fileName, String separator, String enclosure)
      throws IOException {
    return csv(fileName, separator, enclosure, true);
  }

  public static Dataframe csv(String fileName, String separator, String enclosure,
      boolean hasHeader) {
    return new Dataframe(
        CsvFileSource.class,
        fileName,
        separator,
        enclosure,
        hasHeader
    );
  }


}