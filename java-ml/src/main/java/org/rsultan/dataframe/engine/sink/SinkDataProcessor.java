package org.rsultan.dataframe.engine.sink;

import static org.rsultan.utils.Constants.NULL_ROW;

import java.util.concurrent.CountDownLatch;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.queue.QueueFactory;

import java.util.Queue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class SinkDataProcessor<T> extends BaseDataProcessor implements Sink {

  private final Logger LOG = LoggerFactory.getLogger(SinkDataProcessor.class);
  protected final CountDownLatch lock;

  protected SinkDataProcessor() {
    super(QueueFactory.create());
    lock = new CountDownLatch(1);
  }

  @Override
  public void run() {
    try {
      while (isStarted()) {
        final Queue<Row> queue = QueueFactory.get(queueId);
        if (!queue.isEmpty()) {
          var row = queue.poll();
          if (!NULL_ROW.equals(row)) {
            this.consume(row);
          } else {
            stop();
          }
        }
      }
    } catch (Exception e) {
      LOG.error("An unexpected error has occurred", e);
      stop();
    }
  }

  @Override
  public void stop() {
    lock.countDown();
    super.stop();
  }

  public abstract T getResult();

  @Override
  public void feed(Row row) {
    throw new IllegalArgumentException("You cannot feed after a Sink");
  }
}
