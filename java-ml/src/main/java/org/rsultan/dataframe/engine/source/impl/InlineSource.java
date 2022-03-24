package org.rsultan.dataframe.engine.source.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;
import org.rsultan.dataframe.Row;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.rsultan.dataframe.engine.source.SourceDataProcessor;

import static java.util.Optional.ofNullable;
import static org.rsultan.utils.Constants.NULL_ROW;

public class InlineSource extends SourceDataProcessor {

  private final List<List<Object>> rows;

  private AtomicInteger index;

  public InlineSource(String[] header, List<List<Object>> rows) {
    this.header = Arrays.asList(header);
    this.rows = rows.stream().map(ArrayList::new).collect(Collectors.toList());
  }

  @Override
  public void run() {
    super.run();
  }

  @Override
  public Row produce() {
    final int currentIndex = index.getAndIncrement();
    if (currentIndex == rows.size()) {
      return NULL_ROW;
    }
    return new Row(rows.get(currentIndex));
  }

  @Override
  protected boolean canStop(Row row) {
    return NULL_ROW.equals(row);
  }

  @Override
  public void start(ExecutorService executorService) {
    index = new AtomicInteger(0);
    super.start(executorService);
  }
}
