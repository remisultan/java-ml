package org.rsultan.dataframe.engine.sink.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.sink.SinkDataProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RowSink extends SinkDataProcessor<List<Row>> {

  private final Logger LOG = LoggerFactory.getLogger(RowSink.class);
  private final List<Row> rows = new ArrayList<>();

  @Override
  public synchronized void consume(Row row) {
    rows.add(row);
  }

  @Override
  public void start(ExecutorService executorService) {
    super.start(executorService);
  }

  @Override
  public List<Row> getResult() {
    try {
      lock.await();
    } catch (InterruptedException e) {
      LOG.error("Interrupted while waiting for result", e);
    }
    return rows;
  }
}
