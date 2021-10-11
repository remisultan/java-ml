package org.rsultan.dataframe.printer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import org.rsultan.dataframe.Dataframe;

public class DataframeWriter {

  private final Dataframe dataframe;

  private DataframeWriter(Dataframe dataframe) {
    this.dataframe = dataframe;
  }

  public static void write(Dataframe dataframe, String fileName, String separator,
      String enclosure) {
    new DataframeWriter(dataframe).write(new File(fileName), separator, enclosure);
  }


  public void write(File file, String separator, String enclosure) {
    try (BufferedWriter writer = Files.newBufferedWriter(file.toPath())) {
      for (int i = 0; i < dataframe.getRowSize(); i++) {
        List<String> row = new ArrayList<>();
        for (int j = 0; j <  dataframe.getColumnSize(); j++) {
          Object o = dataframe.getColumns()[j].values().get(i);
          if (o instanceof String s) {
            row.add(enclosure + s + enclosure);
          } else {
            row.add(o.toString());
          }
        }
        writer.write(String.join(separator, row) + "\r\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
