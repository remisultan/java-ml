package org.rsultan.dataframe.engine.queue;

import java.util.concurrent.ConcurrentLinkedQueue;
import org.rsultan.dataframe.Row;

import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class QueueFactory {

  private static final Map<String, Queue<Row>> queues = new ConcurrentHashMap<>();

  public static String create() {
    var key = UUID.randomUUID().toString();
    queues.put(key, new ConcurrentLinkedQueue<>());
    return key;
  }

  public static Queue<Row> get(String key) {
    return queues.get(key);
  }

  public static void clear(String key) {
    queues.remove(key);
  }
}

