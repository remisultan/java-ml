package org.rsultan.dataframe.engine.sink.impl;

import static java.util.stream.Collectors.joining;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.sink.SinkDataProcessor;

public class FileSink extends SinkDataProcessor<Void> {

  private final String separator;
  private final String enclosure;
  private final boolean withHeader;
  private final File file;
  private BufferedWriter writer;
  private final AtomicBoolean headerWritten = new AtomicBoolean(false);

  public FileSink(String fileName, String separator, String enclosure, boolean withHeader) {
    this.file = new File(fileName);
    this.separator = separator;

    this.enclosure = enclosure;
    this.withHeader = withHeader;
  }

  @Override
  public synchronized void consume(Row row) {
    try {
      if (withHeader && !headerWritten.get()) {
        writer.write(
            header.stream()
                .map(v -> v instanceof Number ? v.toString() : enclosure + v + enclosure)
                .collect(joining(separator)));
        writer.newLine();
        headerWritten.set(true);
      }
      writer.write(row.values().stream()
          .map(v -> v instanceof Number ? v.toString() : enclosure + v + enclosure)
          .collect(joining(separator)));
      writer.newLine();
    } catch (IOException e) {
      throw new RuntimeException("An unexpected error has occurred " + e.getMessage());
    }
  }

  @Override
  public synchronized void start(ExecutorService executorService) {
    writer = getWriter();
    super.start(executorService);
  }

  @Override
  public void stop() {
    try {
      writer.close();
      super.stop();
    } catch (IOException e) {
      throw new RuntimeException("An unexpected error has occurred " + e.getMessage());
    }
  }

  private BufferedWriter getWriter() {
    try {
      if (file.exists()) {
        Files.delete(file.toPath());
      }
      Files.createFile(file.toPath());
      return Files.newBufferedWriter(file.toPath());
    } catch (Exception e) {
      throw new RuntimeException("An unexpected error has occurred " + e.getMessage());
    }
  }

  @Override
  public Void getResult() {
    throw new UnsupportedOperationException("This operation is not supported");
  }
}
