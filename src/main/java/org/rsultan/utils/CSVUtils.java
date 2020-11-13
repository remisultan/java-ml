package org.rsultan.utils;

import org.rsultan.dataframe.Column;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.function.Function;
import java.util.regex.Pattern;

import static java.util.Collections.singletonList;
import static java.util.stream.IntStream.range;

public class CSVUtils {

    private static final Pattern DOUBLE_VALUE_REGEX = Pattern.compile("\\d+\\.\\d+");
    private static final Pattern LONG_VALUE_REGEX = Pattern.compile("\\d+");

    private static Object getValueWithType(String value) {
        if (DOUBLE_VALUE_REGEX.matcher(value).matches()) {
            return Double.parseDouble(value);
        } else if (LONG_VALUE_REGEX.matcher(value).matches()) {
            return Long.valueOf(value);
        }
        return value;
    }

    public static Column<?>[] read(String fileName, String separator, boolean withHeader) throws IOException {
        var path = Paths.get(fileName);
        var reader = Files.newBufferedReader(path);
        var firstLine = reader.readLine().split(separator);
        var columns = range(0, firstLine.length)
                .boxed()
                .map(buildColumnHeaderName(withHeader, firstLine))
                .toArray(Column[]::new);
        reader.lines().map(line -> line.split(separator)).forEach(lineArray ->
                range(0, lineArray.length).forEach(index -> {
                    var typedValue = getValueWithType(lineArray[index]);
                    columns[index].values().add(typedValue);
                }));
        reader.close();
        return columns;
    }

    private static Function<Integer, Column<?>> buildColumnHeaderName(boolean withHeader, String[] firstLine) {
        return index -> {
            var firstLineCell = firstLine[index];
            return withHeader ?
                    new Column<>(firstLineCell, new ArrayList<>()) :
                    new Column<>("c".concat(index.toString()), new ArrayList<>(singletonList((getValueWithType(firstLineCell)))));
        };
    }
}
