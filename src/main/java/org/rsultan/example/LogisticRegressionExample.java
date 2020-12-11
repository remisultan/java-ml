package org.rsultan.example;

import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

import java.util.List;

public class LogisticRegressionExample {

    public static void main(String[] args) {
        Dataframes
                .create(new Column<>("couleurs", List.of("rouge", "vert", "bleu")))
                .oneHotEncode("couleurs")
                .show(100);
    }
}
