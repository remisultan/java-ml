package org.rsultan.dataframe.engine;

import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import org.rsultan.dataframe.Row;

public interface DataProcessor extends Runnable {

  void feed(Row row);

  void start(ExecutorService executor);

  void stop();

  boolean isStarted();

}
