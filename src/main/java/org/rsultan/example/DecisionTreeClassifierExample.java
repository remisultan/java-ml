package org.rsultan.example;

import static java.lang.Boolean.parseBoolean;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.ENTROPY;

import java.io.IOException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.tree.DecisionTreeClassifier;
import org.rsultan.core.tree.DecisionTreeRegressor;
import org.rsultan.dataframe.Dataframes;

public class DecisionTreeClassifierExample {

  /*
    You can use the iris dataset for classification --> args[0]
    You can use the wine dataset for regression --> args[0]
   */

  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args) throws IOException {
    boolean isClassifier = parseBoolean(args[1]);
    if (isClassifier) {
      classifier(args[0]);
    } else {
      regressor(args[0]);
    }
  }

  private static void regressor(String arg) throws IOException {
    var decisionTreeRegression = new DecisionTreeRegressor(5);
    var dataframe = Dataframes.csv(arg, ";", "\"", true);
    decisionTreeRegression
        .setResponseVariableName("quality")
        .train(dataframe);
    var newDf = decisionTreeRegression.predict(dataframe);
    newDf.show(0, 15000);
  }

  private static void classifier(String arg) throws IOException {
    var decisionTreeClassifier = new DecisionTreeClassifier(5, ENTROPY);
    var dataframe = Dataframes.csv(arg, ",", "\"", false);
    decisionTreeClassifier.setResponseVariableName("c4").train(dataframe);
    var newDf = decisionTreeClassifier.predict(dataframe);
    newDf.show(0, 150);
  }
}
