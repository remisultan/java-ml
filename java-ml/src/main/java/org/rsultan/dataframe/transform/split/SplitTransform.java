package org.rsultan.dataframe.transform.split;

import java.io.Serializable;
import org.rsultan.dataframe.transform.split.SplitDataframe.TrainTestSplit;

public interface SplitTransform extends Serializable {

  TrainTestSplit split();

}
