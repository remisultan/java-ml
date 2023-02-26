package org.rsultan.dataframe.printer;

import static java.lang.String.valueOf;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.IntStream.range;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class DataframePrinter {

  private static final Lock lock = new ReentrantLock();

  public static final String COLUMN_DELIMITER = " ║ ";
  public static final String ROW_LINE_DELIMITER = "=";
  private static final int MAX_DISPLAYABLE_LIST_VAL_CHARS = 10;

  private final Map<?, List<?>> data;
  private final Map<String, Integer> mapMaxSizes;
  private final List<String> mapIndices;

  private DataframePrinter(Map<?, List<?>> data) {
    this.data = data;
    mapMaxSizes = this.data.keySet().stream()
        .map(key -> {
          String keyStr = valueOf(key);
          return Map.entry(keyStr, keyStr.length());
        })
        .collect(toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

    mapIndices = new ArrayList<>(mapMaxSizes.keySet());
  }

  public static DataframePrinter create(Map<?, List<?>> data) {
    return new DataframePrinter(data);
  }

  public void print(int start, int end) {
    var rows = buildRows(start, end);
    var columnsString = computeStringRowWith(mapIndices);
    var dashRow = range(0, columnsString.length()).boxed().map(num -> ROW_LINE_DELIMITER)
        .collect(joining());
    printRows(rows, columnsString, dashRow);
  }

  private List<List<String>> buildRows(int start, int end) {
    return range(0, end - start).boxed().map(num ->
        data.entrySet().stream()
            .map(entry -> Map.entry(valueOf(entry.getKey()), getStringValue(num, entry)))
            .map(entry -> {
              int newLength = entry.getValue().length();
              mapMaxSizes.computeIfPresent(
                  entry.getKey(),
                  (key, oldLength) -> oldLength < newLength ? newLength : oldLength);
              return entry.getValue();
            }).collect(toList())
    ).collect(toList());
  }

  private String getStringValue(Integer num, Entry<?, List<?>> entry) {
    final Object value = entry.getValue().get(num);
    if(value instanceof Collection<?> c && c.size() > MAX_DISPLAYABLE_LIST_VAL_CHARS){
      return valueOf(c.stream().limit(10).collect(toList()))
          .replaceAll("]", ", ...]");
    }
    return valueOf(value);
  }

  private void printRows(List<List<String>> rows, String columnsString, String dashRow) {
    lock.lock();
    try {
      System.out.println(dashRow);
      System.out.println(columnsString);
      System.out.println(dashRow);
      rows.stream().map(this::computeStringRowWith)
          .forEach(strRow -> {
            System.out.println(strRow);
            System.out.println(dashRow);
          });
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      lock.unlock();
    }

  }

  private String computeStringRowWith(List<String> row) {
    return range(0, row.size()).boxed().map(idx -> {
      var colName = mapIndices.get(idx);
      var valMaxSize = mapMaxSizes.get(colName);
      String value = row.get(idx);
      var spaces = range(0, valMaxSize - value.length()).boxed().map(num -> " ").collect(joining());
      return spaces + value;
    }).collect(joining(COLUMN_DELIMITER));
  }
}
