package org.rsultan.dataframe.engine.filter;

import static java.util.Optional.ofNullable;
import static org.rsultan.utils.Constants.NULL_ROW;

import java.util.Queue;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.rsultan.dataframe.engine.queue.QueueFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class FilterDataProcessor extends BaseDataProcessor {

  private final Logger LOG = LoggerFactory.getLogger(FilterDataProcessor.class);

  protected FilterDataProcessor() {
    super(QueueFactory.create());
  }

  @Override
  public void run() {
    try {
      while (isStarted()) {
        final Queue<Row> queue = QueueFactory.get(queueId);
        if (!queue.isEmpty()) {
          final Row row = queue.poll();
          if (!NULL_ROW.equals(row)) {
            if (this.filter(row)) {
              this.feed(row);
            }
          } else {
            stop();
          }
        }
      }
    } catch (Exception e) {
      LOG.error("Error while filtering data", e);
      stop();
    }
  }

  protected abstract boolean filter(Row row);

  @Override
  public void stop() {
    this.feed(NULL_ROW);
    super.stop();
  }
}
