package org.rsultan.dataframe.engine.label;

import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.toCollection;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

public class LabelValueIndexer<T> {

  private final Map<T, Long> indexer = new ConcurrentHashMap<>();
  private final Map<Long, T> reverseIndexer = new ConcurrentHashMap<>();

  @SafeVarargs
  public LabelValueIndexer(T... values) {
    this(Arrays.stream(values).collect(toCollection(TreeSet::new)));
  }

  public LabelValueIndexer(Set<T> values) {
    var iterator = values.iterator();
    Long index = 0L;
    while (iterator.hasNext()) {
      final T value = iterator.next();
      indexer.put(value, index);
      reverseIndexer.put(index, value);
      index++;
    }
  }

  public Long getIndex(Object value) {
    return ofNullable(indexer.get((T) value)).orElse(Long.MIN_VALUE);
  }

  public T getLabelValue(Number index) {
    return reverseIndexer.get(index.longValue());
  }

}
