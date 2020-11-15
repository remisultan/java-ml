package org.rsultan.dataframe;

import org.rsultan.utils.CSVUtils;

import java.io.IOException;

public class Dataframes {

    public static Dataframe create(Column<?>... columns) {
        return new Dataframe(columns);
    }

    public static Dataframe csv(String fileName) throws IOException {
        return csv(fileName, ",");
    }

    public static Dataframe csv(String fileName, String separator) throws IOException {
        return csv(fileName, separator, true);
    }

    public static Dataframe csv(String fileName, String separator, boolean withHeader) throws IOException {
        var columns = CSVUtils.read(fileName, separator, withHeader);
        return new Dataframe(columns);
    }
}
