package org.rsultan.dataframe.engine.source;


import java.util.UUID;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.BaseDataProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class SourceDataProcessor extends BaseDataProcessor implements Source {

  private static final Logger LOG = LoggerFactory.getLogger(SourceDataProcessor.class);
  protected SourceDataProcessor() {
    super(UUID.randomUUID().toString());
  }

  @Override
  public void run() {
    try {
      propagateHeader(getHeader());
      while (isStarted()) {
        final Row row = this.produce();
        this.feed(row);
        if (canStop(row)) {
          stop();
        }
      }
    } catch (Exception e) {
      LOG.error("Error while processing data", e);
      stop();
    }
  }

  protected abstract boolean canStop(Row row);

}
