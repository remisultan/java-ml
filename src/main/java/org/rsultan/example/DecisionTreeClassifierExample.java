package org.rsultan.example;

import static org.rsultan.core.tree.impurity.ImpurityStrategy.ENTROPY;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.tree.DecisionTreeClassifier;
import org.rsultan.dataframe.Dataframes;

public class DecisionTreeClassifierExample {

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    var decisionTreeClassifier = new DecisionTreeClassifier(5, ENTROPY);
    var dataframe = Dataframes.csv(args[0], ",", "\"", false);
    decisionTreeClassifier.setResponseVariableName("c4").train(dataframe);
    var newDf = decisionTreeClassifier.predict(dataframe);
    newDf.show(0, 150);
  }
}
