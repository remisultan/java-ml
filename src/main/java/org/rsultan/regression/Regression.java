package org.rsultan.regression;

import org.rsultan.dataframe.Dataframe;

public interface Regression {

    Regression train(Dataframe dataframe);

    Dataframe predict(Dataframe dataframe);
}
