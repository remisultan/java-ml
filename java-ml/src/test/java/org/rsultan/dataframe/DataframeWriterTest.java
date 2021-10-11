package org.rsultan.dataframe;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.UUID;
import org.junit.jupiter.api.Test;

public class DataframeWriterTest {

  @Test
  public void must_write_dataframe_to_file() throws IOException {
    var headers = new String[]{"c1", "c2", "c3"};
    String filename =
        System.getProperty("java.io.tmpdir") + File.separator + UUID.randomUUID() + ".csv";
    var inputDf = Dataframes.create(headers,
        new Row("1", 2, 3.0D),
        new Row("4", "5", 6),
        new Row(7, 8, 9)
    );
    inputDf.write(filename, ",", "\"");

    File file = new File(filename);
    assertThat(Files.exists(file.toPath())).isTrue();

    var dataframe = Dataframes.csv(file.getAbsolutePath(), ",", "\"", false);
    assertThat(dataframe.getRowSize()).isEqualTo(dataframe.getRowSize());
    assertThat(dataframe.getColumnSize()).isEqualTo(dataframe.getColumnSize());

    Files.delete(file.toPath());
  }
}
