package org.rsultan.dataframe;

import java.io.IOException;
import org.rsultan.utils.CSVUtils;

public class Dataframes {

  public static Dataframe create(Column<?>... columns) {
    return new Dataframe(columns);
  }

  public static Dataframe csv(String fileName) throws IOException {
    return csv(fileName, ",");
  }

  public static Dataframe csv(String fileName, String separator) throws IOException {
    return csv(fileName, separator, "\"", true);
  }

  public static Dataframe csv(String fileName, String separator, String enclosure)
      throws IOException {
    return csv(fileName, separator, enclosure, true);
  }

  public static Dataframe csv(String fileName, String separator, String enclosure,
      boolean withHeader) throws IOException {
    return new Dataframe(
        CSVUtils.read(fileName, separator, enclosure, withHeader)
    );
  }
}
