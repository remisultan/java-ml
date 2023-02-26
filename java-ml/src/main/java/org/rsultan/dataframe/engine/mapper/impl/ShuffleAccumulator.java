package org.rsultan.dataframe.engine.mapper.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.AccumulatorDataProcessor;

public class ShuffleAccumulator extends AccumulatorDataProcessor {

  private List<Row> accumulator;

  @Override
  public Row map(Row element) {
    return element;
  }

  @Override
  protected void accumulate(Row row) {
    accumulator.add(row);
  }

  @Override
  protected void feedFromAccumulator() {
    Collections.shuffle(this.accumulator);
    this.accumulator.stream().map(this::map).forEach(this::feed);
  }

  @Override
  public void start(ExecutorService executorService) {
    accumulator = new ArrayList<>();
    super.start(executorService);
  }

  @Override
  public void stop() {
    accumulator.clear();
    super.stop();
  }
}
