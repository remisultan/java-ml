package org.rsultan.dataframe.engine.mapper;

import static org.rsultan.utils.Constants.NULL_ROW;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.queue.QueueFactory;

public abstract class AccumulatorDataProcessor extends BaseDataProcessor implements
    Mapper<Row, Row> {

  protected List<Row> accumulator;
  private final AtomicBoolean canAccumulate = new AtomicBoolean();

  protected AccumulatorDataProcessor() {
    super(QueueFactory.create());
  }

  @Override
  public void run() {
      while (canAccumulate.get()) {
      final Queue<Row> queue = QueueFactory.get(queueId);
        if (!queue.isEmpty()) {
          final Row row = queue.poll();
          if (NULL_ROW.equals(row)) {
            canAccumulate.set(false);
          } else {
            accumulator.add(row);
          }
        }
      }
      feedFromAccumulator();
    this.stop();
  }

  protected void feedFromAccumulator() {
    this.accumulator.stream().map(this::map).forEach(this::feed);
  }

  @Override
  public void start(ExecutorService executorService) {
    canAccumulate.set(true);
    accumulator = new ArrayList<>();
    super.start(executorService);
  }

  @Override
  public void stop() {
    this.feed(NULL_ROW);
    canAccumulate.set(false);
    super.stop();
  }
}
