package org.rsultan.dataframe;

import java.io.IOException;
import org.rsultan.utils.CSVUtils;

public class Dataframes {

  public static TrainTestDataframe trainTest(Column<?>... columns) {
    return new TrainTestDataframe(create(columns));
  }

  public static TrainTestDataframe trainTest(String[] columnNames, Row... rows) {
    return new TrainTestDataframe(create(columnNames, rows));
  }

  public static Dataframe create(Column<?>... columns) {
    return new Dataframe(columns);
  }

  public static Dataframe create(String[] columnNames, Row... rows) {
    return new Dataframe(columnNames, rows);
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

  public static TrainTestDataframe csvTrainTest(String fileName) throws IOException {
    return csvTrainTest(fileName, ",");
  }

  public static TrainTestDataframe csvTrainTest(String fileName, String separator) throws IOException {
    return csvTrainTest(fileName, separator, "\"");
  }

  public static TrainTestDataframe csvTrainTest(String fileName, String separator, String enclosure)
      throws IOException {
    return csvTrainTest(fileName, separator, enclosure, true);
  }

  public static TrainTestDataframe csvTrainTest(String fileName, String separator, String enclosure,
      boolean withHeader) throws IOException {
    return new TrainTestDataframe(csv(fileName, separator, enclosure, withHeader));
  }

  public static Dataframe csv(String fileName, String separator, String enclosure,
      boolean withHeader) throws IOException {
    return new Dataframe(
        CSVUtils.read(fileName, separator, enclosure, withHeader)
    );
  }
}
