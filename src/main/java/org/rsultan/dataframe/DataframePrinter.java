package org.rsultan.dataframe;

import java.util.List;
import java.util.Map;

import static java.util.Map.Entry.comparingByKey;
import static java.util.stream.Collectors.*;
import static java.util.stream.IntStream.range;

public class DataframePrinter {

    public static final String COLUMN_DELIMITER = " ║ ";
    public static final String ROW_LINE_DELIMITER = "=";

    private final Map<String, List<?>> data;
    private final Map<String, Integer> mapMaxSizes;
    private final List<String> mapIndices;

    private DataframePrinter(Map<String, List<?>> data) {
        this.data = data;
        mapMaxSizes = this.data.keySet().stream()
                .map(key -> Map.entry(key, key.length()))
                .collect(toMap(Map.Entry::getKey, Map.Entry::getValue));

        mapIndices = mapMaxSizes.entrySet()
                .stream().sorted(comparingByKey())
                .map(Map.Entry::getKey)
                .collect(toList());
    }

    public static DataframePrinter create(Map<String, List<?>> data) {
        return new DataframePrinter(data);
    }

    public void print(int number) {
        var rows = buildRows(number);
        var columnsString = computeStringRowWith(mapIndices);
        var dashRow = range(0, columnsString.length()).boxed().map(num -> ROW_LINE_DELIMITER).collect(joining());
        printRows(rows, columnsString, dashRow);
    }

    private List<List<String>> buildRows(int number) {
        return range(0, number).boxed().map(num ->
                data.entrySet().stream()
                        .sorted(comparingByKey())
                        .map(entry -> Map.entry(entry.getKey(), entry.getValue().get(num)))
                        .map(entry -> {
                            var strValue = String.valueOf(entry.getValue());
                            int newLength = strValue.length();
                            mapMaxSizes.computeIfPresent(entry.getKey(), (key, oldLength) -> oldLength < newLength ? newLength : oldLength);
                            return strValue;
                        }).collect(toList())
        ).collect(toList());
    }

    private void printRows(List<List<String>> rows, String columnsString, String dashRow) {
        System.out.println(dashRow);
        System.out.println(columnsString);
        System.out.println(dashRow);
        rows.stream().map(this::computeStringRowWith)
                .forEach(strRow -> {
                    System.out.println(strRow);
                    System.out.println(dashRow);
                });
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
