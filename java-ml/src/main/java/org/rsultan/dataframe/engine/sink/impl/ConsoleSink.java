package org.rsultan.dataframe.engine.sink.impl;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.concurrent.ExecutorService;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.sink.SinkDataProcessor;
import org.rsultan.dataframe.printer.DataframePrinter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static java.util.Objects.nonNull;

public class ConsoleSink extends SinkDataProcessor<Void> {

  private Map<?, List<?>> data;

  private final int start;
  private final int end;

  private final AtomicInteger counter;

  public ConsoleSink(int start, int end) {
    super();
    if (start > end) {
      this.start = Math.min(end, 0);
      this.end = Math.max(start, 0);
    } else {
      this.start = Math.min(start, 0);
      this.end = Math.max(end, 0);
    }
    counter = new AtomicInteger(0);
  }

  @Override
  public void consume(Row row) {
    final int count = counter.getAndIncrement();
    if (count >= start && count <= end) {
      for (int i = 0; i < header.size(); i++) {
        var value = row.values().get(i);
        List<Object> colValues = (List<Object>) data.get(header.get(i));
        if (nonNull(colValues)) {
          colValues.add(value);
        } else {
          var values = new ArrayList<>();
          values.add(value);
          ((Map<Object, List<?>>) data).put(header.get(i), values);
        }
      }
    }
  }

  @Override
  public void start(ExecutorService executorService) {
    data = Collections.synchronizedMap(new LinkedHashMap<>());
    super.start(executorService);
  }

  @Override
  public void stop() {
    super.stop();
    final int size = counter.get();
    int realEnd = Math.min(size, end);
    DataframePrinter.create(data).print(start, realEnd);
  }

  @Override
  public Void getResult() {
    throw new RuntimeException(new IllegalAccessException( "getResult not implemented for ConsoleSink"));
  }
}
